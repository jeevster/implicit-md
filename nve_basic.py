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
import time
from tqdm import tqdm
import pstats
from torchmd.interface import GNNPotentials, PairPotentials, Stack
from torchmd.potentials import ExcludedVolume, LennardJones, LJFamily,  pairMLP
from torchmd.observable import rdf, generate_vol_bins, DifferentiableRDF
import shutil
from utils import radii_to_dists



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
        self.burn_in_frac = params.burn_in_frac
        self.nn = params.nn


        self.cutoff = params.cutoff
        self.gaussian_width = params.gaussian_width
        self.n_width = params.n_width
        self.n_layers = params.n_layers
        self.nonlinear = params.nonlinear
        

        # print("Input parameters")
        # print("Number of particles %d" % self.n_particle)
        # print("Initial temperature %8.8e" % self.temp)
        # print("Box size %8.8e" % self.box)
        # print("epsilon %8.8e" % self.epsilon)
        # print("sigma %8.8e" % self.sigma)
        # print("dt %8.8e" % self.dt)
        # print("Total time %8.8e" % self.t_total)
        # print("Number of steps %d" % self.nsteps)

        # Constant box properties
        self.vol = self.box**3.0
        self.rho = self.n_particle/self.vol

        #GPU
        try:
            self.device = torch.device(torch.cuda.current_device())
        except:
            self.device = "cpu"

        # Define prior potential
        mlp_params = {'n_gauss': int(params.cutoff//params.gaussian_width), 
                'r_start': 0.0,
                'r_end': params.cutoff, 
                'n_width': params.n_width,
                'n_layers': params.n_layers,
                'nonlinear': params.nonlinear}


        self.NN = pairMLP(**mlp_params)

        #prior potential only contains repulsive term
        self.prior = LJFamily(epsilon=params.epsilon, sigma=params.sigma, rep_pow=6, attr_pow=0)

        self.model = Stack({'pairnn': self.NN, 'pair': self.prior}).to(self.device)

        #register backwards hook
        # for module in self.model.modules():
        #     module.register_backward_hook(backward_hook)

        self.model.requires_grad = True
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=5)

        #define differentiable rdf function
        self.diff_rdf = DifferentiableRDF(params, self.device)
        self.diff_rdf_cpu = DifferentiableRDF(params, "cpu")

        #load ground truth rdf
        if self.nn:
            self.results_dir = f"n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}"
            self.gt_rdf = torch.Tensor(np.load(os.path.join('results', self.results_dir, "epoch1_rdf_nn=False.npy"))).to(self.device)


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
                    return torch.Tensor(radius_).to(self.device)

   
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
        return torch.Tensor(velocities).to(self.device)


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
        with torch.enable_grad():
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
            #d = self.dists.detach().numpy()

            #zero out self-interactions
            #dists = (self.dists + 10000000*torch.eye(self.n_particle)).unsqueeze(-1)
        
            if self.nn:
                #energy = self.model(dists).sum()
                energy = self.model(self.dists)
                forces = -compute_grad(inputs=r, output=energy)
            else:
                r2i = (self.sigma/self.dists)**2
                r6i = r2i**3
                energy = 2*self.epsilon*torch.sum(r6i*(r6i - 1))
                #reuse components of potential to calculate virial and forces
                self.internal_virial = -48*self.epsilon*r6i*(r6i - 0.5)/(self.sigma**2)
                forces = -self.internal_virial*r*r2i

        new_forces = torch.zeros((self.dists.shape[0], self.dists.shape[0], 3))
        for i in range(self.dists.shape[0]):
            new_forces[i] = torch.cat(([forces[i, :i], torch.zeros((1,3)).to(self.device), forces[i, i:]]), dim=0)
        f = new_forces.cpu().detach().numpy()

        #import pdb; pdb.set_trace()
        assert(not torch.any(torch.isnan(new_forces)))
        assert self.check_symmetric(f[:, :, 0], mode = 'opposite')
        assert self.check_symmetric(f[:, :, 1], mode = 'opposite')
        assert self.check_symmetric(f[:, :, 2], mode = 'opposite')
       
                
        #sum forces across particles
        return energy, torch.sum(new_forces, axis = 1).to(self.device)
        
       
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
        self.running_dists = torch.cat(self.running_dists).cpu().detach().numpy()
        
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
                self.running_dists.append(self.dists.cpu().detach())
            
         
        # RDF Calculation
        if self.nn: #compute RDF from final frame only
            self.gr = self.diff_rdf(tuple(radii_to_dists(radii, self.box)))
        else: # compute RDF from entire trajectory except for burn-in (do it on CPU to avoid memory issues)
            length = len(self.running_dists)
            self.gr = self.diff_rdf_cpu(self.running_dists[int(self.burn_in_frac*length):])

        #self.hist_gr = self.calc_rdf()
        if self.nn:
            self.optimizer.zero_grad()
            loss = (self.gr - self.gt_rdf).pow(2).mean()
            print(f"Loss: {loss}")

            start = time.time()
            loss.backward()
            end = time.time()

            print("gradient calculation time (s): ",  end - start)
            max_norm = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    norm = torch.linalg.vector_norm(param.grad, dim=-1).max()
                    if  norm > max_norm:
                        max_norm = norm
            print("Max norm: ", max_norm.item())
            
            self.optimizer.step()
            self.scheduler.step(loss)
        
        #logging
        save_dir = f"n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}"
        os.makedirs(os.path.join('results', save_dir), exist_ok = True)
        add = "_nn=False.npy" if not self.nn else ".npy"
        np.save(os.path.join('results', save_dir, f"rdf_epoch{epoch+1}" + add), self.gr.cpu().detach().numpy())
        shutil.copy("config.yaml", os.path.join('results', save_dir))

                
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

    if not params.nn:
        params.n_epochs = 1

    for epoch in range(params.n_epochs):
        print(f"Epoch {epoch+1}")
        simulator.simulate(epoch)
        simulator.reset_system()
    print('Done!')
    
    

