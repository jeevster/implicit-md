import numpy as np
import gsd.hoomd
import torch
import torch.nn as nn
from nff.utils.scatter import compute_grad
from nff.nn.layers import GaussianSmearing
from YParams import YParams
import argparse
import os
from tqdm import tqdm
import pstats
import pdb
from torchmd.interface import GNNPotentials, PairPotentials, Stack
from torchmd.potentials import ExcludedVolume, LennardJones, LJFamily,  pairMLP
from torchmd.observable import rdf, generate_vol_bins, DifferentiableRDF
import torchopt
from torchopt.nn import ImplicitMetaGradientModule
import time
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import shutil
from torch.utils.tensorboard import SummaryWriter
import cProfile



from utils import radii_to_dists, fcc_positions, initialize_velocities, dump_params_to_yml



class ImplicitMDSimulator(ImplicitMetaGradientModule, linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0)):
    def __init__(self, params, model, radii_0, velocities_0, rdf_0):
        super(ImplicitMDSimulator, self).__init__()
        # Initial MD parameters
        np.random.seed(seed=params.seed)
        self.params = params
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
        self.save_intermediate_rdf = params.save_intermediate_rdf
        self.exp_name = params.exp_name

        self.cutoff = params.cutoff
        self.gaussian_width = params.gaussian_width
        self.n_width = params.n_width
        self.n_layers = params.n_layers
        self.nonlinear = params.nonlinear

        self.inference = params.inference
        self.pretrained_model_dir = params.pretrained_model_dir
        
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

        #Register inner parameters
        self.model = model.to(self.device)
        #model.train()
        self.radii = nn.Parameter(radii_0.clone().detach_(), requires_grad=True).to(self.device)
        self.velocities = nn.Parameter(velocities_0.clone().detach_(), requires_grad=True).to(self.device)
        self.rdf = nn.Parameter(rdf_0.clone().detach_(), requires_grad=True).to(self.device)


        #register backwards hook for debugging
        # for module in self.model.modules():
        #     module.register_backward_hook(backward_hook)

        #define differentiable rdf function
        self.diff_rdf = DifferentiableRDF(params, self.device)
        self.diff_rdf_cpu = DifferentiableRDF(params, "cpu")

        

        if self.nn:
            self.save_dir = os.path.join('results', f"IMPLICIT_{self.exp_name}_n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}_dt={self.dt}_ttotal={self.t_total}")
        else: 
            self.save_dir = os.path.join('ground_truth', f"n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}")
        os.makedirs(self.save_dir, exist_ok = True)
        dump_params_to_yml(self.params, self.save_dir)

        #File dump stuff
        self.f = open(f"{self.save_dir}/log.txt", "a+")
        self.t = gsd.hoomd.open(name=f'{self.save_dir}/test_temp{self.temp}.gsd', mode='wb') 
        self.n_dump = params.n_dump # dump for configuration

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
        s = gsd.hoomd.Frame()
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

    def save_checkpoint(self, best=False):
        name = "best_ckpt.pt" if best else "ckpt.pt"
        checkpoint_path = os.path.join(self.save_dir, name)
        torch.save({'model_state': self.model.state_dict()}, checkpoint_path)

    
        
        



    '''CORE MD OPERATIONS'''
    def force_calc(self, radii):
        
        #Get rij matrix
        with torch.enable_grad():
            r = radii.unsqueeze(0) - radii.unsqueeze(1)
            if not r.requires_grad:
                r.requires_grad = True
            #Enforce minimum image convention
            r = -1*torch.where(r > 0.5*self.box, r-self.box, torch.where(r<-0.5*self.box, r+self.box, r))

            #get rid of diagonal 0 entries of r matrix (for gradient stability)
            r = r[~torch.eye(r.shape[0],dtype=bool)].reshape(r.shape[0], -1, 3)
            
            #compute distance matrix:
            dists = torch.sqrt(torch.sum(r**2, axis=2)).unsqueeze(-1)

            #compute energy
            if self.nn:
                energy = self.model(dists)
                forces = -compute_grad(inputs=r, output=energy)
            
            #LJ potential
            else:
                r2i = (self.sigma/dists)**2
                r6i = r2i**3
                energy = 2*self.epsilon*torch.sum(r6i*(r6i - 1))
                #reuse components of potential to calculate virial and forces
                internal_virial = -48*self.epsilon*r6i*(r6i - 0.5)/(self.sigma**2)
                forces = -internal_virial*r*r2i

        #insert 0s back in diagonal entries of force matrix
        # new_forces = torch.zeros((dists.shape[0], dists.shape[0], 3))
        # for i in range(dists.shape[0]):
        #     new_forces[i] = torch.cat(([forces[i, :i], torch.zeros((1,3)).to(self.device), forces[i, i:]]), dim=0)
        # f = new_forces.detach().numpy()

        # #Ensure symmetries
        assert(not torch.any(torch.isnan(forces)))
        # assert self.check_symmetric(f[:, :, 0], mode = 'opposite')
        # assert self.check_symmetric(f[:, :, 1], mode = 'opposite')
        # assert self.check_symmetric(f[:, :, 2], mode = 'opposite')
       
                
        #sum forces across particles
        return energy, torch.sum(forces, axis = 1)#.to(self.device)
        
    
    def forward(self, radii, velocities, forces, calc_rdf = False):
        # Forward process - 1 MD step with Velocity-Verlet integration
        #half-step in velocity
        velocities = velocities + 0.5*self.dt*forces

        #full step in position
        radii = radii + self.dt*velocities

        #PBC correction
        radii = radii/self.box 
        radii = self.box*torch.where(radii-torch.round(radii) >= 0, \
                    (radii-torch.round(radii)), (radii - torch.floor(radii)-1))

        #calculate force at new position
        _, forces = self.force_calc(radii.to(self.device))
        
        #another half-step in velocity
        velocities = (velocities + 0.5*self.dt*forces) 
        #props = self.calc_properties()
        new_rdf = self.rdf
        if calc_rdf:
            new_dists = radii_to_dists(radii, self.box)
            new_rdf = self.diff_rdf(tuple(new_dists.to(self.device))) #calculate the RDF from a single frame
        # try:
        #     diff = (torch.abs(new_rdf - self.calc_rdf)/ self.calc_rdf).mean()
        #     print("mean relative difference in rdf: ", diff)
        #     self.running_diffs.append(diff.item())
        # except:
        #     pass
        #calc_rdf = new_rdf

        # dump frame
        if self.step%self.n_dump == 0:
            #print(self.step, props, file=self.f)
            self.t.append(self.create_frame(frame = self.step/self.n_dump))
            #append dists to running_dists for RDF calculation (remove diagonal entries)
            if not self.nn:
                self.running_dists.append(radii_to_dists(radii, self.box).cpu().detach())
            #np.save(f"inst_rdf_nn/t={self.step}_inst_rdf_n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}_nn={self.nn}.npy", self.calc_rdf.detach().numpy())

        return radii, velocities, forces, new_rdf # return the new distance matrix 



    def optimality(self):
        # Stationary condition construction for calculating implicit gradient
        #print("optimality")
        #Stationarity of the RDF - doesn't change if we do another step of MD
        #Need to compute a residual with respect to every inner param, and each residual has to be the 
        #same shape as the corresponding parameter. 
        #TODO: currently, radii, velocity, and rdf are all params - need to make only rdf a parameter so we can 
        #just compute the rdf residual

        with torch.enable_grad():
            forces = self.force_calc(self.radii)[1]
            new_radii, new_velocities, _, new_rdf = self(self.radii, self.velocities, forces, calc_rdf = True)
        radii_residual  = self.radii - new_radii
        velocity_residual  = self.velocities - new_velocities
        rdf_residual = self.rdf - new_rdf
        
        return (radii_residual, velocity_residual, rdf_residual)


    #top level MD simulation code (i.e the "solver") that returns the optimal "parameter" -aka the equilibriated radii
    def solve(self):
        self.running_dists = []
        
        #Initialize forces/potential of starting configuration
        with torch.no_grad():
            _, forces = self.force_calc(self.radii)
            #Run MD
            print("Start MD trajectory", file=self.f)
            for step in tqdm(range(self.nsteps)):
                self.step = step
                
                calc_rdf = step ==  self.nsteps -1 or (self.save_intermediate_rdf and not self.nn)
                radii, velocities, forces, rdf = self(self.radii, self.velocities, forces, calc_rdf = calc_rdf)
                self.radii.copy_(radii)
                self.velocities.copy_(velocities)
                self.rdf.copy_(rdf)
                if not self.nn and self.save_intermediate_rdf and step % self.n_dump == 0:
                    filename = f"step{step+1}_rdf.npy"
                    np.save(os.path.join(self.save_dir, filename), self.rdf.cpu().detach().numpy())

        #compute ground truth rdf over entire trajectory (do it on CPU to avoid memory issues)
        length = len(self.running_dists)
        save_rdf = self.diff_rdf_cpu(self.running_dists[int(self.burn_in_frac*length):]) if not self.nn else self.rdf

        
        filename ="gt_rdf.npy" if not self.nn else f"rdf_epoch{epoch+1}.npy"

        #pre-pend RDF with name of the pretrained model used in this simulation
        if self.inference and self.pretrained_model_dir:   
            filename = f"{os.path.basename(self.pretrained_model_dir)}_{filename}"
        
        np.save(os.path.join(self.save_dir, filename), save_rdf.cpu().detach().numpy())
        
        
        self.f.close()
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
    parser.add_argument('--epsilon', type=float, default=1.0, help='LJ epsilon')
    parser.add_argument('--sigma', type=float, default=1.0, help='LJ sigma')
    parser.add_argument('--dt', type=float, default=0.005, help='time step for integration')
    parser.add_argument('--dr', type=float, default=0.01, help='bin size for RDF calculation (non-differentiable version)')
    parser.add_argument('--t_total', type=float, default=5, help='total time')
    parser.add_argument('--diameter_viz', type=float, default=0.3, help='particle diameter for Ovito visualization')
    parser.add_argument('--n_dump', type=int, default=10, help='save frequency of configurations (also frequency of frames used for ground truth RDF calculation)')

    parser.add_argument('--save_intermediate_rdf', action = 'store_true', help='Whether to store the RDF along the trajectory for the ground truth')
    parser.add_argument('--burn_in_frac', type=float, default=0.2, help='initial fraction of trajectory to discount when calculating ground truth rdf')
    parser.add_argument('--exp_name', type=str, default = "", help='name of experiment - used as prefix of results folder name')

    #learnable potential stuff
    parser.add_argument('--n_epochs', type=int, default=30, help='number of outer loop training epochs')
    parser.add_argument('--nn', action='store_true', help='use neural network potential')
    parser.add_argument('--pretrained_model_dir', type=str, default= "", help='folder containing pretrained model to initialize with')
    parser.add_argument('--inference', action='store_true', help='just run simulator for one epoch, no training')


    parser.add_argument('--cutoff', type=float, default=2.5, help='LJ cutoff distance')
    parser.add_argument('--gaussian_width', type=float, default=0.1, help='width of the Gaussian used in the RDF')
    parser.add_argument('--n_width', type=int, default=128, help='number of Gaussian functions used in the RDF')
    parser.add_argument('--n_layers', type=int, default=3, help='number of hidden layers in the neural network potential')
    parser.add_argument('--nonlinear', type=str, default='ELU', help='type of nonlinearity used in the neural network potential')


    params = parser.parse_args()

    #GPU
    try:
        device = torch.device(torch.cuda.current_device())
    except:
        device = "cpu"
    
    #Limit CPU usage
    torch.set_num_threads(5)

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

    model = Stack({'nn': NN, 'prior': prior})

    if params.pretrained_model_dir:
        #load checkpoint
        name = "best_ckpt.pt" if params.inference else "ckpt.pt"
        checkpoint_path = os.path.join(params.pretrained_model_dir, name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])

    #inner parameters
    radii_0 = fcc_positions(params.n_particle, params.box, device)
    velocities_0  = initialize_velocities(params.n_particle, params.temp)
    rdf_0  = diff_rdf(tuple(radii_to_dists(radii_0.to(device), params.box)))

    #load ground truth rdf
    if params.nn:
        gt_dir = os.path.join('ground_truth', f"n={params.n_particle}_box={params.box}_temp={params.temp}_eps={params.epsilon}_sigma={params.sigma}")
        gt_rdf = torch.Tensor(np.load(os.path.join(gt_dir, "gt_rdf.npy"))).to(device)
        results_dir = os.path.join('results', f"IMPLICIT_{params.exp_name}_n={params.n_particle}_box={params.box}_temp={params.temp}_eps={params.epsilon}_sigma={params.sigma}_dt={params.dt}_ttotal={params.t_total}")

    #initialize outer loop optimizer/scheduler
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    if not params.nn or params.inference:
        params.n_epochs = 1

    #outer training loop
    losses = []
    grad_times = []
    sim_times = []
    grad_norms = []
    best_outer_loss = 100
    if params.nn:
        writer = SummaryWriter(log_dir = results_dir)
    for epoch in range(params.n_epochs):
        print(f"Epoch {epoch+1}")
        best = False
        #initialize simulator parameterized by a NN model
        simulator = ImplicitMDSimulator(params, model, radii_0, velocities_0, rdf_0)

        optimizer.zero_grad()

        #run MD simulation to get equilibriated radii
        start = time.time()
        equilibriated_simulator = simulator.solve()
        end = time.time()
        sim_time = end-start
        print("MD simulation time (s): ",  sim_time)
        
        #compute RDF loss at the end of the trajectory
        if params.nn:
            outer_loss = (equilibriated_simulator.rdf - gt_rdf).pow(2).mean()
            print("Final RDF Loss: ", outer_loss.item())

            if outer_loss < best_outer_loss:
                best_outer_loss = outer_loss
                best = True
            if not params.inference:
                simulator.save_checkpoint(best = best)

                #compute (implicit) gradient of outer loss wrt model parameters 
                start = time.time()
                torch.autograd.backward(tensors = outer_loss, inputs = list(model.parameters()))
                end = time.time()
                grad_time = end-start
                print("gradient calculation time (s): ",  grad_time)

                max_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        norm = torch.linalg.vector_norm(param.grad, dim=-1).max()
                        if  norm > max_norm:
                            max_norm = norm
                try:
                    print("Max norm: ", max_norm.item())
                except AttributeError:
                    print("Max norm: ", max_norm)

                optimizer.step()
                scheduler.step(outer_loss)
                #log stats
                losses.append(outer_loss.item())
                sim_times.append(sim_time)
                grad_times.append(grad_time)
                grad_norms.append(max_norm.item())

                writer.add_scalar('Loss', losses[-1], global_step=epoch+1)
                writer.add_scalar('Simulation Time', sim_times[-1], global_step=epoch+1)
                writer.add_scalar('Gradient Time', grad_times[-1], global_step=epoch+1)
                writer.add_scalar('Gradient Norm', grad_norms[-1], global_step=epoch+1)
    
    if params.nn and not params.inference:
        stats_write_file = os.path.join(simulator.save_dir, 'stats.txt')
        with open(stats_write_file, "w") as output:
            output.write("Losses: " + str(losses) + "\n")
            output.write("Simulation times: " +  str(sim_times) + "\n")
            output.write("Gradient calculation times: " +  str(grad_times) + "\n")
            output.write("Max gradient norms: " + str(grad_norms))

        writer.close()
    print('Done!')
    
