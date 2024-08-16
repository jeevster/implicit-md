"""
adapted from 
https://github.com/torchmd/mdgrad/tree/master/nff/md
"""
import numpy as np
from ase.md.md import MolecularDynamics
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
import torch
from nff.utils.scatter import compute_grad
from functorch import vmap

def get_stress(positions, velocities, forces, masses, cell):
    """
    Compute the stress tensor for a batch of systems. Useful for NPT simulation
    
    Parameters:
    positions (torch.Tensor): Tensor of shape [B, N, 3] representing positions of atoms.
    velocities (torch.Tensor): Tensor of shape [B, N, 3] representing velocities of atoms.
    forces (torch.Tensor): Tensor of shape [B, N, 3] representing forces on atoms.
    masses (torch.Tensor): Tensor of shape [1, N, 1] representing masses of atoms.
    cell (torch.Tensor) : Tensor of shape [B, 3, 3] representing the cell matrix.
    
    Returns:
    torch.Tensor: Stress tensor of shape [B, 3, 3].
    """
    B, N, _ = positions.shape
    V = torch.linalg.det(cell).unsqueeze(-1).unsqueeze(-1)

    # Potential (Virial) Contribution
    # wrap positions before virial calculation
    diag = vmap(torch.diag)(cell).unsqueeze(1)
    wrapped_positions = ((positions / diag) % 1) * diag  - diag/2
    outer_product = wrapped_positions.unsqueeze(-1) * forces.unsqueeze(-2)
    stress_tensor_potential = outer_product.sum(dim=1) / (3*V)

    # Kinetic Contribution
    kinetic_contrib = (masses.unsqueeze(-1) * velocities.unsqueeze(-1) * velocities.unsqueeze(-2)).sum(dim=1) / V

    # Total Stress Tensor
    stress_tensor = stress_tensor_potential + kinetic_contrib
    
    return stress_tensor


class NoseHoover(MolecularDynamics):
    def __init__(self,
                 atoms,
                 model,
                 timestep,
                 temperature,
                 ttime,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 device = "cpu",
                 **kwargs):

        super().__init__(
                         atoms,
                         timestep,
                         trajectory,
                         logfile,
                         loginterval)

        # Initialize simulation parameters

        # Q is chosen to be 6 N kT
        self.device = device
        self.dt = timestep
    
        self.Natom = atoms.get_number_of_atoms()
        self.T = temperature
        self.targeEkin = 0.5 * (3.0 * self.Natom) * self.T
        self.ttime = ttime  # * units.fs
        self.Q = 3.0 * self.Natom * self.T * (self.ttime * self.dt)**2
        self.zeta = torch.Tensor([0.0]).to(self.device)
        self.model = model
        self.z = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device)
        self.masses = torch.Tensor(self.atoms.get_masses().reshape(-1, 1)).to(self.device)

    def step(self, radii, velocities):

        # get current acceleration and velocity:
        with torch.enable_grad():
            energy = self.model(pos = radii, z = self.z)
        forces = -compute_grad(inputs = radii, output = energy)

        
        accel = forces / self.masses

        # make full step in position
        radii = radii + vel * self.dt + \
            (accel - self.zeta * vel) * (0.5 * self.dt ** 2)
        self.atoms.set_positions(x)

        # record current velocities
        KE_0 = self.atoms.get_kinetic_energy()

        # make half a step in velocity
        vel_half = vel + 0.5 * self.dt * (accel - self.zeta * vel)
        self.atoms.set_velocities(vel_half)

        # make a full step in accelerations
        with torch.enable_grad():
            pos = torch.Tensor(self.atoms.get_positions()).to(self.device).requires_grad_(True)
            energy = self.model(pos = pos, z = self.z)
        forces = -compute_grad(inputs = pos, output = energy)
        accel = forces / self.atoms.get_masses().reshape(-1, 1)

        # make a half step in self.zeta
        self.zeta = self.zeta + 0.5 * self.dt * \
            (1/self.Q) * (KE_0 - self.targeEkin)

        # make another halfstep in self.zeta
        self.zeta = self.zeta + 0.5 * self.dt * \
            (1/self.Q) * (self.atoms.get_kinetic_energy() - self.targeEkin)

        # make another half step in velocity
        vel = (self.atoms.get_velocities() + 0.5 * self.dt * accel) / \
            (1 + 0.5 * self.dt * self.zeta)
        self.atoms.set_velocities(vel)


        return self.atoms.get_positions(), self.atoms.get_velocities()


class NoseHooverChain(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 ttime,
                 num_chains,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 **kwargs):

        super().__init__(
                         atoms,
                         timestep,
                         trajectory,
                         logfile,
                         loginterval)

        # Initialize simulation parameters

        self.dt = timestep

        self.N_dof = 3*atoms.get_number_of_atoms()
        self.T = temperature

        # in units of fs:
        self.ttime = ttime
        self.Q = 2 * np.array([self.N_dof * self.T * (self.ttime * self.dt)**2,
                           *[self.T * (self.ttime * self.dt)**2]*(num_chains-1)])
        self.targeEkin = 1/2 * self.N_dof * self.T

        # self.zeta = np.array([0.0]*num_chains)
        self.p_zeta = np.array([0.0]*num_chains)

    def get_zeta_accel(self):

        p0_dot = 2 * (self.atoms.get_kinetic_energy() - self.targeEkin)- \
            self.p_zeta[0]*self.p_zeta[1] / self.Q[1]
        p_middle_dot = self.p_zeta[:-2]**2 / self.Q[:-2] - \
            self.T - self.p_zeta[1:-1] * self.p_zeta[2:]/self.Q[2:]
        p_last_dot = self.p_zeta[-2]**2 / self.Q[-2] - self.T
        p_dot = np.array([p0_dot, *p_middle_dot, p_last_dot])

        return p_dot / self.Q

    def half_step_v_zeta(self):

        v = self.p_zeta / self.Q
        accel = self.get_zeta_accel()
        v_half = v + 1/2 * accel * self.dt
        return v_half

    def half_step_v_system(self):

        v = self.atoms.get_velocities()
        accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1) 
        accel -= v * self.p_zeta[0] / self.Q[0]
        v_half = v + 1/2 * accel * self.dt
        return v_half

    def full_step_positions(self):

        accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1)
        new_positions = self.atoms.get_positions() + self.atoms.get_velocities() * self.dt + \
            (accel - self.p_zeta[0] / self.Q[0])*(self.dt)**2
        return new_positions

    def step(self):

        new_positions = self.full_step_positions()
        self.atoms.set_positions(new_positions)

        v_half_system = self.half_step_v_system()
        v_half_zeta = self.half_step_v_zeta()

        self.atoms.set_velocities(v_half_system)
        self.p_zeta = v_half_zeta * self.Q

        v_full_zeta = self.half_step_v_zeta()
        accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1)
        v_full_system = (v_half_system + 1/2 * accel * self.dt) / \
            (1 + 0.5 * self.dt * v_full_zeta[0])

        self.atoms.set_velocities(v_full_system)
        self.p_zeta = v_full_zeta * self.Q