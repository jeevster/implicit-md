import numpy as np
import gsd.hoomd
import math
import torch
from nff.utils.scatter import compute_grad
from nff.nn.layers import GaussianSmearing
from scipy.stats import maxwell
from YParams import YParams
import argparse
import os
from tqdm import tqdm
import cProfile as profile
import pstats
import pdb
from torchmd.interface import GNNPotentials, PairPotentials, Stack
from torchmd.potentials import ExcludedVolume, LennardJones, LJFamily,  pairMLP
from torchmd.observable import rdf, generate_vol_bins

def backward_hook(module, grad_input, grad_output):
    print('Module:', module)
    print('Grad Input:', grad_input)
    print('Grad Output:', grad_output)

class DifferentiableRDF(torch.nn.Module):
    def __init__(self, params):
        super(DifferentiableRDF, self).__init__()
        start = 0
        range =  params.box #torch.max(self.running_dists)
        nbins = int(range/params.dr)


        V, vol_bins, bins = generate_vol_bins(start, range, nbins, dim=3)

        self.V = V
        self.vol_bins = vol_bins
        #self.device = system.device
        self.bins = bins

        self.smear = GaussianSmearing(
            start=start,
            stop=bins[-1],
            n_gaussians=nbins,
            width=params.gaussian_width,
            trainable=False
        )

    def forward(self, running_dists):
        running_dists = torch.cat(running_dists)
        count = self.smear(running_dists.reshape(-1).squeeze()[..., None]).sum(0) 
        norm = count.sum()   # normalization factor for histogram 
        count = count / norm   # normalize 
        gr =  count / (self.vol_bins / self.V )  
        return gr



class MDSimulator:
    def __init__(self, params):
        # Initial parameters
        np.random.seed(seed=params.seed)
        self.n_particle = params.n_particle
        self.temp = params.temp
        self.kbt0 = params.kbt0
        self.box = params.box
        self.diameter_viz = params.diameter_viz
        self.epsilon = params.epsilon
        self.sigma = params.sigma
        self.dt = params.dt
        self.t_total = params.t_total
        self.dr = params.dr
        self.nsteps = np.rint(self.t_total/self.dt).astype(np.int32)
        self.nn = params.nn

        self.cutoff = params.cutoff
        self.gaussian_width = params.gaussian_width
        self.n_width = params.n_width
        self.n_layers = params.n_layers
        self.nonlinear = params.nonlinear

        print("Input parameters")
        print("Number of particles %d" % self.n_particle)
        print("Initial temperature %8.8e" % self.temp)
        print("Box size %8.8e" % self.box)
        print("epsilon %8.8e" % self.epsilon)
        print("sigma %8.8e" % self.sigma)
        print("dt %8.8e" % self.dt)
        print("Total time %8.8e" % self.t_total)
        print("Number of steps %d" % self.nsteps)

        # Constant box properties
        self.vol = self.box**3.0
        self.rho = self.n_particle/self.vol

        # Define prior potential
        mlp_params = {'n_gauss': int(params.cutoff//params.gaussian_width), 
                'r_start': 0.0,
                'r_end': params.cutoff, 
                'n_width': params.n_width,
                'n_layers': params.n_layers,
                'nonlinear': params.nonlinear}


        self.NN = pairMLP(**mlp_params)

        #prior potential only contains repulsive term
        self.prior = LJFamily(epsilon=params.epsilon, sigma=params.sigma, rep_pow=12, attr_pow=4)

        self.model = Stack({'pairnn': self.NN, 'pair': self.prior})

        #register backwards hook
        # for module in self.model.modules():
        #     module.register_backward_hook(backward_hook)

        self.model.requires_grad = True
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=1e-6)

        #define differentiable rdf function
        self.diff_rdf = DifferentiableRDF(params)

        #load ground truth rdf
        self.gt_rdf = torch.Tensor(np.load(f"rdf_n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}_nn=False.npy"))


        self.reset_system()
        
        
        

    def reset_system(self):
        #Initialize positions
        self.radii = self.fcc_positions()
        self.radii -= self.box/2 # convert to -L/2 to L/2 space for ease with PBC
        assert(torch.max(self.radii) <=self.box/2 and torch.min(self.radii) >= -1*self.box/2) #assert normalization conditions
        self.running_dists = []

        #initialize velocities
        self.velocities = self.initialize_velocities()

        #Initialize forces/potential of starting configuration
        self.potential, self.forces = self.force_calc()
        #import pdb; pdb.set_trace()

        #File dump stuff
        self.t = gsd.hoomd.open(name='test.gsd', mode='wb') 
        self.n_dump = params.n_dump # dump for configuration
        
        #check if inital momentum is zero
        initial_props = self.calc_properties()
        self.f = open("log.txt", "a+")
        print("Initial: ", initial_props, file = self.f)


    # Initialize configuration
    # Radii
    # FCC lattice
    def fcc_positions(self):
        from itertools import product
        # round-up to nearest fcc box
        cells = np.ceil((self.n_particle/4.0)**(1.0/3.0)).astype(np.int32) #cells in each dimension (assume 4 particles per unit cell)
        self.cell_size = self.box/cells 
        radius_ = np.empty((self.n_particle,3)) #initial positions of particles
        r_fcc = np.array ([[0.25,0.25,0.25],[0.25,0.75,0.75],[0.75,0.75,0.25],[0.75,0.25,0.75]], dtype=np.float64)
        i = 0
        for ix, iy, iz in product(list(range(cells)),repeat=3): # triple loop over unit cells
            for a in range(4): # 4 atoms in a unit cell
                radius_[i,:] = r_fcc[a,:] + np.array([ix,iy,iz]).astype(np.float64) # 0..nc space
                radius_[i,:] = radius_[i,:]*self.cell_size#/self.box # normalize to [0,1]
                i = i+1
                if(i==self.n_particle): #  break when we have n_particle in our box
                    return torch.Tensor(radius_)

   
    # Procedure to initialize velocities
    def initialize_velocities(self):
        
        vel_dist = maxwell()
        velocities = vel_dist.rvs(size = (self.n_particle, 3))
        #shift so that initial momentum is zero
        velocities -= np.mean(velocities, axis = 0)

        #scale velocities to match desired temperature
        sum_vsq = np.sum(np.square(velocities))
        p_dof = 3*(self.n_particle-1)
        correction_factor = math.sqrt(p_dof*self.temp/sum_vsq)
        velocities *= correction_factor
        return torch.Tensor(velocities)


    def check_symmetric(self, a, mode, tol=1e-4):
        if mode == 'opposite':
            return np.all(np.abs(a + a.T) < tol)
        
        return np.all(np.abs(a - a.T) < tol)

    def force_calc(self):
        #TODO
        # Evaluate forces
        # Using LJ potential
        # u_lj(r_ij) = 4*epsilon*[(sigma/r_ij)^12-(sigma/r_ij)^6]
        # You can get energy and the pressure for free out of this calculation if you do it right
        
         #Get rij matrix
        r = self.radii.unsqueeze(0) - self.radii.unsqueeze(1)
        
        #Enforce minimum image convention
        r = -1*torch.where(r > 0.5*self.box, r-self.box, torch.where(r<-0.5*self.box, r+self.box, r))
        r = r[~torch.eye(r.shape[0],dtype=bool)].reshape(r.shape[0], -1, 3)
        try:
            r.requires_grad = True
        except RuntimeError:
            pass
        #compute distance matrix:
        self.dists = torch.sqrt(torch.sum(r**2, axis=2)).unsqueeze(-1)
        d = self.dists.detach().numpy()

        #zero out self-interactions
        #dists = (self.dists + 10000000*torch.eye(self.n_particle)).unsqueeze(-1)
       
        if self.nn:
            #energy = self.model(dists).sum()
            energy = self.model(self.dists)
            forces = -compute_grad(inputs=r, output=energy)
            new_forces = torch.zeros((self.dists.shape[0], self.dists.shape[0], 3))
            for i in range(self.dists.shape[0]):
                new_forces[i] = torch.cat(([forces[i, :i], torch.zeros((1,3)), forces[i, i:]]), dim=0)
    
        else:
            r2i = (self.sigma/dists)**2
            r6i = r2i**3
            energy = 2*self.epsilon*torch.sum(r6i*(r6i - 1))
            #reuse components of potential to calculate virial and forces
            self.internal_virial = -48*self.epsilon*r6i*(r6i - 0.5)/(self.sigma**2)
            forces = -self.internal_virial*r*r2i
        
        f = new_forces.detach().numpy()

        #import pdb; pdb.set_trace()
        assert(not torch.any(torch.isnan(new_forces)))
        assert self.check_symmetric(f[:, :, 0], mode = 'opposite')
        assert self.check_symmetric(f[:, :, 1], mode = 'opposite')
        assert self.check_symmetric(f[:, :, 2], mode = 'opposite')
       
                
        #sum forces across particles
        return energy, torch.sum(new_forces, axis = 1)
        
       
    # Function to dump simulation frame that is readable in Ovito
    # Also stores radii and velocities in a compressed format which is nice
    def create_frame(self, frame):
        # Particle positions, velocities, diameter
            
        radii = self.radii.detach()
        partpos = radii.tolist()
        velocities = self.velocities.detach().tolist()
        diameter = self.diameter_viz*self.sigma*np.ones((self.n_particle,))
        diameter = diameter.tolist()

        # Now make gsd file
        s = gsd.hoomd.Snapshot()
        s.configuration.step = frame
        s.particles.N=self.n_particle
        s.particles.position = partpos
        s.particles.velocity = velocities
        s.particles.diameter = diameter
        s.configuration.box=[self.box,self.box,self.box,0,0,0]
        return s


    def calc_properties(self):
        #TODO
        # Calculate properties of interest in this function
        p_dof = 3*self.n_particle-3
        vel_squared = torch.sum(torch.square(self.velocities))
        ke = vel_squared/2
        temp = 2*ke/p_dof
        #w = -1/6*torch.sum(self.internal_virial)
        #pressure = w/self.vol + self.rho*self.kbt0
        pressure = torch.Tensor(0)
        return {"Temperature": temp.item(),
                "Pressure": pressure,
                "Total Energy": (ke+self.potential).item(),
                "Momentum Magnitude": torch.norm(torch.sum(self.velocities, axis =0)).item()}

    def calc_rdf(self):
        #Calculate RDF histogram
        self.running_dists = torch.cat(self.running_dists).numpy()
        
        range = np.max(self.running_dists)
        

        freqs, bins = np.histogram(self.running_dists, bins=int(range/self.dr), range = (0, range))
        #normalize
        n_log = self.nsteps/self.n_dump
        freqs = np.float64(freqs/(self.n_particle*n_log))
        

        #compute ideal gas equivalent
        r = np.arange(len(bins))*self.dr
        freqs_id = 4*math.pi*self.rho/3 * ((r+self.dr)**3 - r**3)[0:-1]

        gr = freqs/freqs_id
        

        return gr

    #top level MD simulation code
    def simulate(self, epoch):
    
        print("Start MD trajectory", file=self.f)

        # Velocity Verlet integrator
        for step in tqdm(range(self.nsteps)):
            
            self.velocities = self.velocities + 0.5*self.dt*self.forces
            #import pdb; pdb.set_trace()
            self.radii = self.radii + self.dt*self.velocities
            #PBC
            self.radii /= self.box
            self.radii = self.box*torch.where(self.radii-torch.round(self.radii) >= 0, \
                        (self.radii-torch.round(self.radii)), (self.radii - torch.floor(self.radii)-1))
                               
            self.potential, self.forces = self.force_calc()
            
            self.velocities = self.velocities + 0.5*self.dt*self.forces
            
            props = self.calc_properties()

            # dump frame
            if step%self.n_dump == 0:
                print(step, props, file=self.f)
                self.t.append(self.create_frame(frame = step/self.n_dump))
                #append dists to running_dists for RDF calculation (remove diagonal entries)
                self.running_dists.append(self.dists)
            
         
        # RDF Calculation
        self.gr = self.diff_rdf(self.running_dists)
        self.optimizer.zero_grad()
        loss = (self.gr - self.gt_rdf).pow(2).mean()
        print(f"Loss: {loss}")
        
        #loss.backward()
        # max_norm = 0
        # for param in self.model.parameters():
        #     norm = torch.linalg.vector_norm(param.grad, dim=-1).max()
        #     if  norm > max_norm:
        #         max_norm = norm
        # print("Max norm: ", max_norm.item())
        
        #self.optimizer.step()
            
        np.save(f"epoch{epoch+1}_rdf_n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}_nn={self.nn}.npy", self.gr.detach().numpy())

                
        self.f.close()            

if __name__ == "__main__":

    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='config.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    #initialize simulator
    simulator = MDSimulator(params)

    for epoch in range(1):
        print(f"Epoch {epoch+1}")
        simulator.simulate(epoch)
        simulator.reset_system()
    print('Done!')
    
    

