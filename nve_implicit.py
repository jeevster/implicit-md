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
from torchmd.observable import rdf, generate_vol_bins, DifferentiableRDF
import torchopt
from torchopt.nn import ImplicitMetaGradientModule

from utils import radii_to_dists


def backward_hook(module, grad_input, grad_output):
    print('Module:', module)
    print('Grad Input:', grad_input)
    print('Grad Output:', grad_output)


class ImplicitMDSimulator(ImplicitMetaGradientModule, linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0)):
    def __init__(self, params, model):
        super(ImplicitMDSimulator, self).__init__()
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

        #Initialize pair potentials model - this gets registered as a "meta-parameter" since it was passed in as input to init
        self.model = model
        self.model.requires_grad = True

        #register backwards hook for debugging
        # for module in self.model.modules():
        #     module.register_backward_hook(backward_hook)

        #define differentiable rdf function
        self.diff_rdf = DifferentiableRDF(params)

        #load ground truth rdf
        if self.nn:
            self.results_dir = f"n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}"
            self.gt_rdf = torch.Tensor(np.load(os.path.join('results', self.results_dir, "epoch1_rdf_" + self.results_dir + "_nn=False.npy")))

        #reset positions and velocities to initial FCC lattice 
        self.reset_system()
        

    def reset_system(self):
        #Initialize positions
        self.radii = self.fcc_positions()
        self.radii.requires_grad = True
        self.radii = self.radii - self.box/2 # convert to -L/2 to L/2 space for ease with PBC
        assert(torch.max(self.radii) <=self.box/2 and torch.min(self.radii) >= -1*self.box/2) #assert normalization conditions
        self.running_dists = []

        #initialize velocities
        self.velocities = self.initialize_velocities()
        self.velocities.requires_grad = True

        #Initialize forces/potential of starting configuration
        self.potential = torch.zeros((1, ))
        self.forces = torch.zeros((self.n_particle, 3))
        self.potential.requires_grad = True
        self.forces.requires_grad = True
        self.potential, self.forces = self.force_calc()

        #File dump stuff
        self.t = gsd.hoomd.open(name='test.gsd', mode='wb') 
        self.n_dump = params.n_dump # dump for configuration
        
        #check if inital momentum is zero
        #initial_props = self.calc_properties()
        self.f = open("log.txt", "a+")
        #print("Initial: ", initial_props, file = self.f)


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


    '''CORE MD OPERATIONS'''
    def force_calc(self):
        
        #Get rij matrix
        import pdb; pdb.set_trace()
        r = self.radii.unsqueeze(0) - self.radii.unsqueeze(1)
        try:
            r.requires_grad = True
        except:
            pass
        #gradients getting detached here for some reason
       
        #Enforce minimum image convention
        r = -1*torch.where(r > 0.5*self.box, r-self.box, torch.where(r<-0.5*self.box, r+self.box, r))

        #get rid of diagonal 0 entries of r matrix (for gradient stability)
        r = r[~torch.eye(r.shape[0],dtype=bool)].reshape(r.shape[0], -1, 3)
        
        #compute distance matrix:
        if hasattr(self, 'dists'):
            self.dists.copy_(torch.sqrt(torch.sum(r**2, axis=2)).unsqueeze(-1))
        else:
            self.dists = torch.sqrt(torch.sum(r**2, axis=2)).unsqueeze(-1)

        #compute energy
        if self.nn:
            energy = self.model(self.dists)
            forces = -compute_grad(inputs=r, output=energy) #getting assertion error here since r.requires_grad = False for some reason
        else:
            r2i = (self.sigma/self.dists)**2
            r6i = r2i**3
            energy = 2*self.epsilon*torch.sum(r6i*(r6i - 1))
            #reuse components of potential to calculate virial and forces
            self.internal_virial = -48*self.epsilon*r6i*(r6i - 0.5)/(self.sigma**2)
            forces = -self.internal_virial*r*r2i

        #insert 0s back in diagonal entries of force matrix
        new_forces = torch.zeros((self.dists.shape[0], self.dists.shape[0], 3))
        for i in range(self.dists.shape[0]):
            new_forces[i] = torch.cat(([forces[i, :i], torch.zeros((1,3)), forces[i, i:]]), dim=0)
        f = new_forces.detach().numpy()

        #Ensure symmetries
        assert(not torch.any(torch.isnan(new_forces)))
        assert self.check_symmetric(f[:, :, 0], mode = 'opposite')
        assert self.check_symmetric(f[:, :, 1], mode = 'opposite')
        assert self.check_symmetric(f[:, :, 2], mode = 'opposite')
       
                
        #sum forces across particles
        return energy, torch.sum(new_forces, axis = 1)
        
    
    def forward(self):
        # Forward process - 1 MD step with Velocity-Verlet integration
        #half-step in velocity
        self.velocities = self.velocities + 0.5*self.dt*self.forces

        #full step in position
        import pdb; pdb.set_trace()
        self.radii.copy_(self.radii + self.dt*self.velocities)

        #PBC correction
        self.radii.copy_(self.radii/self.box)
        
        self.radii.copy_(self.box*torch.where(self.radii-torch.round(self.radii) >= 0, \
                    (self.radii-torch.round(self.radii)), (self.radii - torch.floor(self.radii)-1)))

        #calculate force
        self.potential, self.forces = self.force_calc()
        
        
        #another half-step in velocity
        self.velocities = (self.velocities + 0.5*self.dt*self.forces) 
        props = self.calc_properties()

        # dump frame
        if step%self.n_dump == 0:
            print(step, props, file=self.f)
            self.t.append(self.create_frame(frame = step/self.n_dump))
            #append dists to running_dists for RDF calculation (remove diagonal entries)
            self.running_dists.append(self.dists)
        
        new_dists = radii_to_dists(self.radii, self.box)
        self.calc_rdf = self.diff_rdf(new_dists) #calculate the RDF
        return new_dists # return the new distance matrix 

    def optimality(self):
        # Stationary condition construction for calculating implicit gradient
        
        #Stationarity of the RDF - doesn't change if we do another step of MD
        return (self.diff_rdf(self.forward()) - self.diff_rdf(self.dists)).pow(2).mean()

    #Question: should we instead define objective function and let torchopt detect optimality conditions?
    
    #top level MD simulation code (i.e the "solver") that returns the optimal "parameter" -aka the equilibriated radii
    def solve(self, epoch):
        #Run MD
        print("Start MD trajectory", file=self.f)
        for step in tqdm(range(self.nsteps)):
            self.forward()
        
        np.save(f"epoch{epoch+1}_rdf_n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}_nn={self.nn}.npy", self.gr.detach().numpy())
        self.f.close()
        return self

    
if __name__ == "__main__":

    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='config.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    #initialize model
    mlp_params = {'n_gauss': int(params.cutoff//params.gaussian_width), 
                'r_start': 0.0,
                'r_end': params.cutoff, 
                'n_width': params.n_width,
                'n_layers': params.n_layers,
                'nonlinear': params.nonlinear}


    NN = pairMLP(**mlp_params)

    #prior potential only contains repulsive term
    prior = LJFamily(epsilon=params.epsilon, sigma=params.sigma, rep_pow=6, attr_pow=0)

    model = Stack({'nn': NN, 'prior': prior})

    #initialize simulator
    simulator = ImplicitMDSimulator(params, model)

    #initialize optimizer/scheduler
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    

    #outer training loop
    for epoch in range(1):
        print(f"Epoch {epoch+1}")
        optimizer.zero_grad()

        #run MD simulation to get equilibriated radii
        optimal_simulator = simulator.solve(epoch)

        #compute RDF loss at the end of the trajectory
        outer_loss = (optimal_simulator.calc_rdf - optimal_simulator.gt_rdf).pow(2).mean()

        #compute (implicit gradient of loss wrt model parameters)
        print(torch.autograd.grad(outer_loss, simulator.model.parameters())) # dL/dtheta

        optimizer.step()
        import pdb; pdb.set_trace()

        #reset system back to initial state
        simulator.reset_system()
    print('Done!')
    
    

