import numpy as np
import gsd.hoomd
import math
from scipy.stats import maxwell
from YParams import YParams
import argparse
import os
from tqdm import tqdm
import cProfile as profile
import pstats
import pdb



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

        # note: generally easy to keep units for radius and velocities in box=1 units 
        # and specifically -0.5 to 0.5 space

        #Initialize positions
        self.radii = self.fcc_positions()
        self.radii -= self.box/2 # convert to -L/2 to L/2 space for ease with PBC
        assert(np.max(self.radii) <=self.box/2 and np.min(self.radii) >= -1*self.box/2) #assert normalization conditions
        self.running_dists = []

        #initialize velocities
        self.velocities = self.initialize_velocities()

        #Initialize forces/potential of starting configuration
        self.potential, self.forces = self.force_calc()

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
                    return radius_

   
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
        return velocities


    def check_symmetric(self, a, tol=1e-8):
        return np.all(np.abs(a-a.T) < tol)

    def force_calc(self):
        #TODO
        # Evaluate forces
        # Using LJ potential
        # u_lj(r_ij) = 4*epsilon*[(sigma/r_ij)^12-(sigma/r_ij)^6]
        # You can get energy and the pressure for free out of this calculation if you do it right
        
         #Get rij matrix
        r = np.expand_dims(self.radii, axis=0) - np.expand_dims(self.radii, axis=1)
        
        #Enforce minimum image convention
        r = -1*np.where(r > 0.5*self.box, r-self.box, np.where(r<-0.5*self.box, r+self.box, r))
        
        #compute distance matrix:
        self.dists = np.sqrt(np.sum(r**2, axis=2))
        #assert self.check_symmetric(self.dists)
        
        #print(np.min(dists + 100*np.identity(self.n_particle)))
        #problem: min of dists is going to zero, which is causing undefined forces and thus undefined radii
        #initial forces are ~ 10^6 and are getting larger and larger (10^17 by the 3rd timestep) since dists are collapsing
        #Main question: why are dists collapsing? Error with PBC implementation? Or issue with initial forces
        
        #assert(np.max(self.dists) <= self.box*math.sqrt(3)/2) #assert max possible distance with PBC

        #zero out self-interactions
        dists = np.expand_dims(self.dists + 10000000*np.identity(self.n_particle), axis = -1) 
       
        
        #compute potential and forces
        r2i = (self.sigma/dists)**2
        
        r6i = r2i**3
        potential = 2*self.epsilon*np.sum(r6i*(r6i - 1))

        #reuse components of potential to calculate virial and forces
        self.internal_virial = -48*self.epsilon*r6i*(r6i - 0.5)/(self.sigma**2)
        forces = -self.internal_virial*r*r2i
        #sum forces across particles
        return potential, np.sum(forces, axis = 1)
        
       
    # Function to dump simulation frame that is readable in Ovito
    # Also stores radii and velocities in a compressed format which is nice
    def create_frame(self, frame):
        # Particle positions, velocities, diameter
            
        radii = self.radii
        partpos = radii.tolist()
        velocities = self.velocities.tolist()
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
        vel_squared = np.sum(np.square(self.velocities))
        ke = vel_squared/2
        temp = 2*ke/p_dof
        w = -1/6*np.sum(self.internal_virial)
        pressure = w/self.vol + self.rho*self.kbt0
        return {"Temperature": temp,
                "Pressure": pressure,
                "Total Energy": ke+self.potential,
                "Momentum Magnitude": np.linalg.norm(np.sum(self.velocities, axis =0))}

    def calc_rdf(self):
        #Calculate RDF histogram
        self.running_dists = np.concatenate(self.running_dists)
        
        range = np.max(self.running_dists)
        

        freqs, bins = np.histogram(self.running_dists, bins=int(range/self.dr), range = (0, range))
        #normalize
        n_log = self.nsteps/self.n_dump
        freqs = np.float64(freqs/(self.n_particle*n_log))
        

        #compute ideal gas equivalent
        r = np.arange(len(bins))*self.dr
        freqs_id = 4*math.pi*self.rho/3 * ((r+self.dr)**3 - r**3)[0:-1]

        gr = freqs/freqs_id
        np.save(f"rdf_n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}.npy", gr)
        return gr


    #top level MD simulation code
    def simulate(self):
    

        # Open file to dump log file
        
        print("Start MD trajectory", file=self.f)

        # NVE integration
        # Equilibration
        for step in tqdm(range(self.nsteps)):
            
            # TODO: Velocity Verlet algorithm
            
            self.velocities = self.velocities + 0.5*self.dt*self.forces
            self.radii = self.radii + self.dt*self.velocities
            #PBC
            self.radii /= self.box
            self.radii = self.box*np.where(self.radii-np.round(self.radii) >= 0, \
                        (self.radii-np.round(self.radii)), (self.radii - np.floor(self.radii)-1))
                               
            self.potential, self.forces = self.force_calc()
            
            self.velocities = self.velocities + 0.5*self.dt*self.forces
            
            props = self.calc_properties()

            # dump frame
            if step%self.n_dump == 0:
                print(step, props, file=self.f)
                self.t.append(self.create_frame(frame = step/self.n_dump))
                #append dists to running_dists for RDF calculation (remove diagonal entries)
                self.running_dists.append(self.dists[~np.eye(self.dists.shape[0],dtype=bool)].reshape(self.dists.shape[0],-1))

            
        # TODO
        # Things left to do    
        # RDF Calculation
        self.gr = self.calc_rdf()
        # I recommend analyzing diffusion coefficient from the trajectories you dump    
                
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

    #run simulation
    simulator.simulate()
    print('Done!')
    
    

