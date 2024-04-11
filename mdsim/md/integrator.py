"""
adapted from 
https://github.com/torchmd/mdgrad/tree/master/nff/md
"""
import torch
from mdsim.common.registry import registry
from ase import units


@registry.register_integrator("NoseHoover")
class NoseHoover:
    def __init__(self, calculator, masses, n_replicas, n_atoms, config, device):
        self.device = device
        self.calculator = calculator
        self.masses = masses
        self.n_replicas = n_replicas
        self.n_atoms = n_atoms
        self.dt = config["timestep"] * units.fs
        self.temp = config["temperature"]
        print(f"Simulation Temperature: {self.temp}")
        self.temp *= units.kB
        self.targeEkin = 0.5 * (3.0 * self.n_atoms) * self.temp
        self.ttime = config["integrator_config"]["ttime"]
        self.Q = 3.0 * self.n_atoms * self.temp * (self.ttime * self.dt) ** 2
        self.zeta = torch.zeros((self.n_replicas, 1, 1)).to(self.device)

    def step(self, radii, velocities, forces, zeta, retain_grad=False):
        # get current accelerations
        accel = forces / self.masses
        # make full step in position
        radii = (
            radii
            + velocities * self.dt
            + (accel - zeta * velocities) * (0.5 * self.dt**2)
        )
        # record current KE
        KE_0 = (
            1
            / 2
            * (self.masses * torch.square(velocities)).sum(axis=(1, 2), keepdims=True)
        )
        # make half a step in velocity
        velocities = velocities + 0.5 * self.dt * (accel - zeta * velocities)
        # make a full step in accelerations
        energy, forces = self.calculator.calculate_energy_force(radii, retain_grad)
        accel = forces / self.masses
        # make a half step in self.zeta
        zeta = zeta + 0.5 * self.dt * (1 / self.Q) * (KE_0 - self.targeEkin)
        # get updated KE
        ke = (
            1
            / 2
            * (self.masses * torch.square(velocities)).sum(axis=(1, 2), keepdims=True)
        )
        # make another halfstep in self.zeta
        zeta = zeta + 0.5 * self.dt * (1 / self.Q) * (ke - self.targeEkin)
        # make another half step in velocity
        velocities = (velocities + 0.5 * self.dt * accel) / (1 + 0.5 * self.dt * zeta)

        return radii, velocities, energy, forces, zeta


@registry.register_integrator("Langevin")
class Langevin:
    def __init__(self, calculator, masses, n_replicas, n_atoms, config, device):
        self.device = device
        self.calculator = calculator
        self.masses = masses
        self.n_atoms = n_atoms
        self.dt = config["timestep"] * units.fs
        self.temp = config["temperature"]
        print(f"Simulation Temperature: {self.temp}")
        self.temp *= units.kB
        self.gamma = config["integrator_config"]["gamma"] / (1000 * units.fs)
        self.noise_f = (
            (2.0 * self.gamma / self.masses * self.temp * self.dt)
            .sqrt()
            .to(self.device)
        )

    def step(self, radii, velocities, forces, retain_grad=False):
        # full step in position
        radii = radii + self.dt * velocities
        # calculate force at new position
        energy, forces = self.calculator.calculate_energy_force(radii, retain_grad)
        noise = torch.randn_like(velocities)
        # full step in velocities
        velocities = (
            velocities
            + self.dt * (forces / self.masses - self.gamma * velocities)
            + self.noise_f * noise
        )
        return radii, velocities, energy, forces, noise
