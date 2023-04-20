import numpy as np
import torch
import torch.nn as nn
from nff.utils.scatter import compute_grad
import argparse
import os
from tqdm import tqdm
import random
from torchmd.interface import Stack
from torchmd.potentials import LJFamily,  pairMLP
from torchmd.observable import DifferentiableRDF
import torchopt
from torchopt.nn import ImplicitMetaGradientModule
from sys import getrefcount
from functorch import vmap
from utils import radii_to_dists, fcc_positions, initialize_velocities, \
                    dump_params_to_yml, powerlaw_inv_cdf, print_active_torch_tensors

class ImplicitMDSimulator(ImplicitMetaGradientModule, linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0)):
    def __init__(self, params, model, radii_0, velocities_0, rdf_0):
        super(ImplicitMDSimulator, self).__init__()
        
        self.params = params
        self.n_particle = params.n_particle
        self.temp = params.temp
        self.kbt0 = params.kbt0
        self.box = params.box
        self.pbc = not params.no_pbc
        self.dt = params.dt
        self.t_total = params.t_total

        #Nose-Hoover Thermostat stuff
        self.nvt_time = params.nvt_time
        self.targeEkin = 0.5 * (3.0 * self.n_particle) * self.temp
        self.Q = 3.0 * self.n_particle * self.temp * (50 * self.dt)**2 #found that this setting works well


        self.diameter_viz = params.diameter_viz
        self.epsilon = params.epsilon
        self.rep_power = params.rep_power
        self.attr_power = params.attr_power
        self.poly = params.poly
        self.poly_power = params.poly_power
        self.min_sigma = params.min_sigma
        self.sigma = params.sigma
        
        self.dr = params.dr
        self.n_nvt_steps = np.rint(self.nvt_time/self.dt).astype(np.int32)
        self.nsteps = np.rint(self.t_total/self.dt).astype(np.int32)
        self.burn_in_frac = params.burn_in_frac
        self.nn = params.nn
        self.n_replicas = params.n_replicas
        self.save_intermediate_rdf = params.save_intermediate_rdf
        self.exp_name = params.exp_name

        self.diffusion_window = params.diffusion_window

        self.cutoff = params.cutoff
        self.gaussian_width = params.gaussian_width
        self.n_width = params.n_width
        self.n_layers = params.n_layers
        self.nonlinear = params.nonlinear

        self.c_0 = -28*self.epsilon / (self.cutoff**12)
        self.c_2 = 48*self.epsilon / (self.cutoff**14)
        self.c_4 = -21*self.epsilon / (self.cutoff**16)

        # Constant box properties
        self.vol = self.box**3.0
        self.rho = self.n_particle/self.vol

        #GPU
        try:
            self.device = torch.device(torch.cuda.current_device())
        except:
            self.device = "cpu"

        self.zeros = torch.zeros((1, 1, 3)).to(self.device)
        
        #limit CPU usage
        torch.set_num_threads(10)

        #Register inner parameters
        self.model = model#.to(self.device)
        self.radii = nn.Parameter(radii_0.detach_(), requires_grad=True).to(self.device)
        self.velocities = nn.Parameter(velocities_0.detach_(), requires_grad=True).to(self.device)
        self.rdf = nn.Parameter(rdf_0.detach_(), requires_grad=True).to(self.device)
        
        #get per-particle sigmas
        u = torch.rand(self.n_particle)
        self.particle_sigmas = powerlaw_inv_cdf(u, power = self.poly_power, y_min = self.min_sigma).clip(max = 1/0.45 * self.min_sigma)
        sigma1 = self.particle_sigmas.unsqueeze(0)
        sigma2 = self.particle_sigmas.unsqueeze(1)
        self.sigma_pairs = 1/2 * (sigma1 + sigma2)*(1 - self.epsilon*torch.abs(sigma1 - sigma2))
        self.sigma_pairs = self.sigma_pairs[~torch.eye(self.sigma_pairs.shape[0],dtype=bool)].reshape(1, self.sigma_pairs.shape[0], -1, 1).to(self.device)
        self.particle_sigmas = self.particle_sigmas
    
        #define vectorized differentiable rdf  functions
        self.diff_rdf = vmap(DifferentiableRDF(params, self.device), -1)
        self.diff_rdf_cpu = vmap(DifferentiableRDF(params, "cpu"), -1)
        
        if self.nn:
            add = "polylj_" if self.poly else ""
            self.save_dir = os.path.join('results', f"IMPLICIT_{add}_{self.exp_name}_n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}_dt={self.dt}_ttotal={self.t_total}")
        else: 
            add = "_polylj" if self.poly else ""
            self.save_dir = os.path.join('ground_truth' + add, f"n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}")


        os.makedirs(self.save_dir, exist_ok = True)
        dump_params_to_yml(self.params, self.save_dir)


    '''CORE MD OPERATIONS'''
    def force_calc(self, radii):

        with torch.enable_grad():
            #Get rij matrix
            r = radii.unsqueeze(-3) - radii.unsqueeze(-2)
            if not r.requires_grad:
                r.requires_grad = True

            #Enforce minimum image convention
            r = -1*torch.where(r > 0.5*self.box, r-self.box, torch.where(r<-0.5*self.box, r+self.box, r))

            #get rid of diagonal 0 entries of r matrix (for gradient stability)
            r = r[:, ~torch.eye(r.shape[1],dtype=bool)].reshape(r.shape[0], r.shape[1], -1, 3)
            
            #compute distance matrix:
            dists = torch.sqrt(torch.sum(r**2, axis=-1)).unsqueeze(-1)

            #compute energy
            if self.nn:
                energy = self.model((dists/self.sigma_pairs)) if self.poly else self.model(dists)
                forces = -compute_grad(inputs=r, output=energy)
                
            #LJ potential
            else:
                parenth = (self.sigma_pairs/dists)
                rep_term = parenth ** self.rep_power
                energy = torch.sum(rep_term + self.c_0 + self.c_2*parenth**-2 + self.c_4*parenth**-4)
                forces = -compute_grad(inputs=r, output=energy)
                
            #calculate which forces to keep
            keep = dists/self.sigma_pairs <= self.cutoff if self.poly else dists/self.sigma <= self.cutoff
            #apply cutoff
            forces = torch.where(~keep, self.zeros, forces)
            # #Ensure no Nans
            assert(not torch.any(torch.isnan(forces))) 
            #sum forces across particles
            return energy, torch.sum(forces, axis = -2)

    def forward_nvt(self, radii, velocities, forces, zeta, calc_rdf = False):
        # get current acceleration
        accel = forces

        # make full step in position
        radii = radii + velocities * self.dt + \
            (accel - zeta * velocities) * (0.5 * self.dt ** 2)

        #PBC correction
        if self.pbc:
            radii = radii/self.box 
            radii = self.box*torch.where(radii-torch.round(radii) >= 0, \
                        (radii-torch.round(radii)), (radii - torch.floor(radii)-1))

        # record current velocities
        KE_0 = torch.sum(torch.square(velocities), axis = (1,2), keepdims=True) / 2
        
        # make half a step in velocity
        velocities = velocities + 0.5 * self.dt * (accel - zeta * velocities)

        # make a full step in accelerations
        energy, forces = self.force_calc(radii.to(self.device))
        accel = forces

        # make a half step in self.zeta
        zeta = zeta + 0.5 * self.dt * (1/self.Q) * (KE_0 - self.targeEkin)

        #get updated KE
        ke = torch.sum(torch.square(velocities), axis = (1,2), keepdims=True) / 2

        # make another halfstep in self.zeta
        zeta = zeta + 0.5 * self.dt * \
            (1/self.Q) * (ke - self.targeEkin)

        # make another half step in velocity
        velocities = (velocities + 0.5 * self.dt * accel) / \
            (1 + 0.5 * self.dt * zeta)

        if calc_rdf:
            new_dists = radii_to_dists(radii, self.params)
            new_dists = new_dists / self.sigma_pairs

        new_rdf = self.diff_rdf(tuple(new_dists.to(self.device).permute((1, 2, 3, 0)))) if calc_rdf else 0 #calculate the RDF from a single frame
        return radii, velocities, forces, zeta, new_rdf
        
    
    def optimality(self):
        # Stationary condition construction for calculating implicit gradient
        
        with torch.enable_grad():
            #get current forces
            forces = self.force_calc(self.radii)[1]
            #make an MD step
            new_radii, new_velocities, _, _, new_rdf = self.forward_nvt(self.radii, self.velocities, forces, self.zeta, calc_rdf = True)

        radii_residual  = self.radii - new_radii
        velocity_residual  = self.velocities - new_velocities
        rdf_residual = self.rdf - new_rdf
        return (radii_residual, velocity_residual, rdf_residual)


    #top level MD simulation code (i.e the "solver") that returns the optimal "parameter" -aka the equilibriated radii
    def solve(self):
        #Initialize forces/potential of starting configuration
        with torch.no_grad():
            _, forces = self.force_calc(self.radii)
            zeta = torch.zeros((self.n_replicas, 1, 1)).to(self.device)
            #Run MD
            print("Start MD trajectory")
            
            for step in tqdm(range(self.nsteps)):
                self.step = step
                calc_rdf = step ==  self.nsteps -1 or (self.save_intermediate_rdf and not self.nn)
                radii, velocities, forces, zeta, rdf = self.forward_nvt(self.radii, self.velocities, forces, zeta, calc_rdf = calc_rdf)
                self.radii.copy_(radii)
                self.velocities.copy_(velocities)
                self.rdf.copy_(rdf)
            self.zeta = zeta   
        return self

    
if __name__ == "__main__":

    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='config.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument('--n_particle', type=int, default=256, help='number of particles')
    parser.add_argument('--temp', type=float, default=1, help='temperature in reduced units')
    parser.add_argument('--seed', type=int, default=123, help='random seed used to initialize velocities')
    parser.add_argument('--kbt0', type=float, default=1.8, help='multiplier for pressure calculation')
    parser.add_argument('--box', type=int, default=7, help='box size')
    parser.add_argument('--no_pbc', action='store_true', help='do not do periodic boundary condition')

    parser.add_argument('--epsilon', type=float, default=1.0, help='LJ epsilon')
    parser.add_argument('--rep_power', type=float, default=12, help='LJ repulsive exponent')
    parser.add_argument('--attr_power', type=float, default=0, help='LJ attractive exponent')
    parser.add_argument('--poly', action='store_true', help='vary sigma (diameter) of each particle according to power law')
    parser.add_argument('--poly_power', type=float, default = -3.0, help='power of power law')
    parser.add_argument('--min_sigma', type=float, default = 0.5, help='minimum sigma for power law distribution')
    parser.add_argument('--sigma', type=float, default=1.0, help='LJ sigma for non-poly systems')
    parser.add_argument('--dt', type=float, default=0.005, help='time step for integration')
    parser.add_argument('--dr', type=float, default=0.01, help='bin size for RDF calculation (non-differentiable version)')
    parser.add_argument('--rdf_sample_frac', type=float, default=1, help='fraction of particles to sample for RDF')
    parser.add_argument('--diffusion_window', type=int, default=100, help='number of timesteps on which to fit linear regression for estimating diffusion coefficient')

    parser.add_argument('--t_total', type=float, default=5, help='total time')
    parser.add_argument('--nvt_time', type=float, default=5, help='time for NVT equilibration')

    parser.add_argument('--diameter_viz', type=float, default=0.3, help='particle diameter for Ovito visualization')
    parser.add_argument('--n_dump', type=int, default=10, help='save frequency of configurations (also frequency of frames used for ground truth RDF calculation)')
    parser.add_argument('--save_intermediate_rdf', action = 'store_true', help='Whether to store the RDF along the trajectory for the ground truth')
    parser.add_argument('--burn_in_frac', type=float, default=0.2, help='initial fraction of trajectory to discount when calculating ground truth rdf')
    parser.add_argument('--exp_name', type=str, default = "", help='name of experiment - used as prefix of results folder name')

    #learnable potential stuff
    parser.add_argument('--n_epochs', type=int, default=30, help='number of outer loop training epochs')
    parser.add_argument('--n_replicas', type=int, default=5, help='number of simulations to parallelize over')

    parser.add_argument('--nn', action='store_true', help='use neural network potential')
    parser.add_argument('--cutoff', type=float, default=2.5, help='LJ cutoff distance')
    parser.add_argument('--gaussian_width', type=float, default=0.1, help='width of the Gaussian used in the RDF')
    parser.add_argument('--n_width', type=int, default=128, help='number of Gaussian functions used in the RDF')
    parser.add_argument('--n_layers', type=int, default=3, help='number of hidden layers in the neural network potential')
    parser.add_argument('--nonlinear', type=str, default='ELU', help='type of nonlinearity used in the neural network potential')

    parser.add_argument('--rdf_loss_weight', type=float, default=1, help='coefficient in front of RDF loss term')
    parser.add_argument('--diffusion_loss_weight', type=float, default=100, help='coefficient in front of diffusion coefficient loss term')


    params = parser.parse_args()

    #GPU
    try:
        device = torch.device(torch.cuda.current_device())
    except:
        device = "cpu"

    #Set random seeds
    np.random.seed(seed=params.seed)
    torch.manual_seed(params.seed)
    random.seed(params.seed)

    #initialize RDF calculator
    diff_rdf = DifferentiableRDF(params, device)

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

    model = Stack({'nn': NN, 'prior': prior}).to(device)
    radii_0 = fcc_positions(params.n_particle, params.box, device).unsqueeze(0).repeat(params.n_replicas, 1, 1)
    velocities_0  = initialize_velocities(params.n_particle, params.temp, n_replicas = params.n_replicas)
    rdf_0  = diff_rdf(tuple(radii_to_dists(radii_0[0].unsqueeze(0).to(device), params))).unsqueeze(0).repeat(params.n_replicas, 1)

    #load ground truth rdf and diffusion coefficient
    if params.nn:
        add = "_polylj" if params.poly else ""
        gt_dir = os.path.join('ground_truth' + add, f"n={params.n_particle}_box={params.box}_temp={params.temp}_eps={params.epsilon}_sigma={params.sigma}")
        gt_rdf = torch.Tensor(np.load(os.path.join(gt_dir, "gt_rdf.npy"))).to(device)
        gt_diff_coeff = torch.Tensor(np.load(os.path.join(gt_dir, "gt_diff_coeff.npy"))).to(device)
        add = "polylj_" if params.poly else ""
        results_dir = os.path.join('results', f"IMPLICIT_{add}_{params.exp_name}_n={params.n_particle}_box={params.box}_temp={params.temp}_eps={params.epsilon}_sigma={params.sigma}_dt={params.dt}_ttotal={params.t_total}")

    #initialize outer loop optimizer
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)

    if not params.nn:
        params.n_epochs = 1

    for epoch in range(params.n_epochs):
        
        print(f"Epoch {epoch+1}")
        #initialize simulator parameterized by a NN model
        simulator = ImplicitMDSimulator(params, model, radii_0, velocities_0, rdf_0)

        optimizer.zero_grad()

        #run MD simulation to get equilibriated radii
        equilibriated_simulator = simulator.solve()
        
        #compute loss at the end of the trajectory
        if params.nn:
            rdf_loss = (equilibriated_simulator.rdf - gt_rdf).pow(2).mean()
            outer_loss = params.rdf_loss_weight*rdf_loss 
            torch.autograd.backward(tensors = outer_loss, inputs = list(model.parameters()))
            optimizer.step()

        print_active_torch_tensors()
    
    print('Done!')
    
    

