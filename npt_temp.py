import sys
import weakref
from typing import IO, Optional, Tuple, Union

import numpy as np

from ase import Atoms, units
from ase.md.md import MolecularDynamics

import torch
from torch import Tensor
linalg = torch.linalg

from functorch import vmap

# Delayed imports:  If the trajectory object is reading a special ASAP version
# of HooverNPT, that class is imported from Asap.Dynamics.NPTDynamics.


class NPT:

    classname = "NPT"  # Used by the trajectory.
    _npt_version = 2   # Version number, used for Asap compatibility.

    def __init__(
        self,
        atoms_batch,
        radii,
        velocities,
        masses,
        cell,
        pbc, 
        atomic_numbers,
        energy_force_func,
        timestep: float,
        temperature: Optional[float] = None,
        externalstress: Optional[float] = None,
        ttime: Optional[float] = None,
        pfactor: Optional[float] = None,
        *,
        temperature_K: Optional[float] = None,
        mask: Optional[Union[Tuple[int], Tensor]] = None,
    ):
        
        self.dt = timestep
        self.atoms_batch = atoms_batch
        self.pbc = pbc
        self.atomic_numbers = atomic_numbers
        self.radii = radii
        self.velocities = velocities
        self.masses = masses
        self.cell = cell
        self.energy_force_func = energy_force_func
        self.device = self.radii.device
        if externalstress is None and pfactor is not None:
            raise TypeError("Missing 'externalstress' argument.")
        self.zero_center_of_mass_momentum(verbose=1)
        self.set_temperature(temperature = temperature)
        if externalstress is not None:
            self.set_stress(externalstress)
        self.set_mask(mask)
        self.eta = torch.zeros((self._getnreplicas(), 3, 3), dtype=torch.float32).to(self.device)
        self.zeta = torch.zeros((self._getnreplicas(), 1, 1), dtype=torch.float32).to(self.device)
        self.zeta_integrated = torch.zeros((self._getnreplicas(), 1, 1), dtype=torch.float32).to(self.device)
        self.initialized = 0
        self.ttime = ttime
        self.pfactor_given = pfactor
        self._calculateconstants()
        self.timeelapsed = 0.0
        self.frac_traceless = 1

        self.initialize()

    def set_temperature(self, temperature=None, *, temperature_K=None):
        """Set the temperature.

        Parameters:

        temperature: float (deprecated)
            The new temperature in eV.  Deprecated, use ``temperature_K``.

        temperature_K: float (keyword-only argument)
            The new temperature, in K.
        """
        self.temperature = temperature
        self._calculateconstants()

    def set_stress(self, stress):
        """Set the applied stress.

        Must be a symmetric 3x3 tensor, a 6-vector representing a symmetric
        3x3 tensor, or a number representing the pressure.

        Use with care, it is better to set the correct stress when creating
        the object.
        """

        if isinstance(stress, (int, float)) or (isinstance(stress, Tensor) and stress.numel() == 1):
            stress = torch.tensor([-stress, -stress, -stress, 0.0, 0.0, 0.0]).to(self.device)
        else:
            stress = torch.tensor(stress).to(self.device)
            if stress.shape == (3, 3):
                if not self._issymmetric(stress):
                    raise ValueError(
                        "The external stress must be a symmetric tensor.")
                stress = torch.tensor((stress[0, 0], stress[1, 1],
                                   stress[2, 2], stress[1, 2],
                                   stress[0, 2], stress[0, 1])).to(self.device)
            elif stress.shape != (6,):
                raise ValueError("The external stress has the wrong shape.")
        self.externalstress = stress


    def set_mask(self, mask):
        """Set the mask indicating dynamic elements of the computational box.

        If set to None, all elements may change.  If set to a 3-vector
        of ones and zeros, elements which are zero specify directions
        along which the size of the computational box cannot change.
        For example, if mask = (1,1,0) the length of the system along
        the z-axis cannot change, although xz and yz shear is still
        possible.  May also be specified as a symmetric 3x3 array indicating
        which strain values may change.

        Use with care, as you may "freeze in" a fluctuation in the strain rate.
        """
        if mask is None:
            mask = torch.ones((3,)).to(self.device)
        if not hasattr(mask, "shape"):
            mask = torch.tensor(mask).to(self.device)
        if mask.shape != (3,) and mask.shape != (3, 3):
            raise RuntimeError('The mask has the wrong shape ' +
                               '(must be a 3-vector or 3x3 matrix)')
        else:
            mask = torch.ne(mask, 0)  # Make sure it is 0/1

        if mask.shape == (3,):
            self.mask = torch.outer(mask, mask)
        else:
            self.mask = mask


    def set_fraction_traceless(self, fracTraceless):
        """set what fraction of the traceless part of the force
        on eta is kept.

        By setting this to zero, the volume may change but the shape may not.
        """
        self.frac_traceless = fracTraceless


    def get_strain_rate(self):
        """Get the strain rate as an upper-triangular 3x3 matrix.

        This includes the fluctuations in the shape of the computational box.

        """
        return torch.tensor(self.eta, copy=1).to(self.device)


    def set_strain_rate(self, rate):
        """Set the strain rate.  Must be an upper triangular 3x3 matrix.

        If you set a strain rate along a direction that is "masked out"
        (see ``set_mask``), the strain rate along that direction will be
        maintained constantly.
        """
        if not (rate.shape == (3, 3) and self._isuppertriangular(rate)):
            raise ValueError("Strain rate must be an upper triangular matrix.")
        self.eta = rate
        if self.initialized:
            # Recalculate h_past and eta_past so they match the current value.
            self._initialize_eta_h()


    def get_time(self):
        "Get the elapsed time."
        return self.timeelapsed


    def run(self, steps):
        """Perform a number of time steps."""
        if not self.initialized:
            self.initialize()
        else:
            if self.have_the_atoms_been_changed():
                raise NotImplementedError(
                    "You have modified the atoms since the last timestep.")

        for _ in range(steps):
            self.step(retain_grad=False)
            self.nsteps += 1
            self.call_observers()


    def have_the_atoms_been_changed(self):
        "Checks if the user has modified the positions or momenta of the atoms"
        limit = 1e-10
        h = self._getbox()
        if max(abs((h - self.h).ravel())) > limit:
            self._warning("The computational box has been modified.")
            return 1
        expected_r = torch.bmm(self.q + 0.5, h)
        err = max(abs((expected_r - self.radii).ravel()))
        if err > limit:
            self._warning("The atomic positions have been modified: " +
                          str(err))
            return 1
        return 0

    def step(self, retain_grad = False):
        """Perform a single time step.

        Assumes that the forces and stresses are up to date, and that
        the positions and momenta have not been changed since last
        timestep.
        """

        # Assumes the following variables are OK
        # q_past, q, q_future, p, eta, eta_past, zeta, zeta_past, h, h_past
        #
        # q corresponds to the current positions
        # p must be equal to self.atoms.GetCartesianMomenta()
        # h must be equal to self.atoms.GetUnitCell()
        #
        # print "Making a timestep"
        dt = self.dt
        h_future = self.h_past + 2 * dt * torch.bmm(self.h, self.eta)
        if self.pfactor_given is None:
            deltaeta = torch.zeros(6, dtype=torch.float32).to(self.device)
        else:
            stress = self.get_stress()
            deltaeta = -2 * dt * ((self.pfact * linalg.det(self.h)).unsqueeze(-1) *
                                  (stress - self.externalstress))

        if self.frac_traceless == 1:
            eta_future = self.eta_past + self.mask * \
                self._makeuppertriangular(deltaeta)
        else:
            trace_part, traceless_part = self._separatetrace(
                self._makeuppertriangular(deltaeta))
            eta_future = (self.eta_past + trace_part +
                          self.frac_traceless * traceless_part)

        deltazeta = 2 * dt * self.tfact * (self.get_kinetic_energy() -
                                           self.desiredEkin)
        zeta_future = self.zeta_past + deltazeta
        # Advance time
        self.timeelapsed += dt
        self.h_past = self.h
        self.h = h_future
        self.inv_h = linalg.inv(self.h)
        self.q_past = self.q
        self.q = self.q_future
        self._setbox_and_positions(self.h, self.q)
        self.eta_past = self.eta
        self.eta = eta_future
        self.zeta_past = self.zeta
        self.zeta = zeta_future
        self.zeta_integrated += dt * self.zeta
        force = self.get_forces(retain_grad)
        self._calculate_q_future(force)
        self.velocities = torch.bmm(self.q_future - self.q_past, self.h / (2 * dt))

        return self.radii, self.velocities, self.get_forces(), self._getbox()
        

    def get_stress(self):
        
        """
        Compute the stress tensor for a batch of systems. Useful for NPT simulation
        
        Parameters:
        positions (torch.Tensor): Tensor of shape [B, N, 3] representing positions of atoms.
        velocities (torch.Tensor): Tensor of shape [B, N, 3] representing velocities of atoms.
        forces (torch.Tensor): Tensor of shape [B, N, 3] representing forces on atoms.
        masses (torch.Tensor): Tensor of shape [1, N, 1] representing masses of atoms.
        volume (float): Volume of the system.
        
        Returns:
        torch.Tensor: Stress tensor of shape [B, 3, 3].
        """
        B, N, _ = self.radii.shape
        V = self._getvolume().unsqueeze(-1).unsqueeze(-1)

        # Potential (Virial) Contribution
        outer_product = self.radii.unsqueeze(-1) * self.get_forces().unsqueeze(-2)
        stress_tensor_potential = outer_product.sum(dim=1) / V

        # Kinetic Contribution
        kinetic_contrib = (self.masses.unsqueeze(-1) * self.velocities.unsqueeze(-1) * self.velocities.unsqueeze(-2)).sum(dim=1) / V

        # Total Stress Tensor
        stress_tensor = stress_tensor_potential + kinetic_contrib

        # Convert from 3x3 matrix to six-vector
        stress_tensor = self._makesixvector(stress_tensor)
        
        return stress_tensor

    def initialize(self):
        """Initialize the dynamics.

        The dynamics requires positions etc for the two last times to
        do a timestep, so the algorithm is not self-starting.  This
        method performs a 'backwards' timestep to generate a
        configuration before the current.

        This is called automatically the first time ``run()`` is called.
        """
        print("Initializing the NPT dynamics.")

        dt = self.dt
        self.h = self._getbox()
        if not self._isuppertriangular(self.h):
            print("I am", self)
            print("self.h:")
            print(self.h)
            print("Min:", min((self.h[1, 0], self.h[2, 0], self.h[2, 1])))
            print("Max:", max((self.h[1, 0], self.h[2, 0], self.h[2, 1])))
            raise NotImplementedError(
                "Can (so far) only operate on lists of atoms where the "
                "computational box is an upper triangular matrix.")
        self.inv_h = linalg.inv(self.h)
        # The contents of the q arrays should migrate in parallel simulations.
        # self._make_special_q_arrays()
        self.q = torch.bmm(self.radii, self.inv_h) - 0.5
        # zeta and eta were set in __init__
        self._initialize_eta_h()
        deltazeta = dt * self.tfact * (self.get_kinetic_energy() -
                                       self.desiredEkin)
        self.zeta_past = self.zeta - deltazeta
        self._calculate_q_past_and_future()
        self.initialized = 1

    def get_potential_energy(self):
        energy, _ = self.energy_force_func(self.radii)
        return energy.squeeze()
    
    def get_forces(self, retain_grad = False):
        _, force = self.energy_force_func(self.radii, retain_grad = retain_grad)
        return force
    
    def get_kinetic_energy(self):
        return 0.5 * (self.masses*torch.square(self.velocities)).sum(axis = (1,2), keepdims=True)


    def get_gibbs_free_energy(self):
        """Return the Gibb's free energy, which is supposed to be conserved.

        Requires that the energies of the atoms are up to date.

        This is mainly intended as a diagnostic tool.  If called before the
        first timestep, Initialize will be called.
        """
        if not self.initialized:
            self.initialize()
        n = self._getnatoms()
        # tretaTeta = sum(diagonal(matrixmultiply(transpose(self.eta),
        #                                        self.eta)))
    
        contractedeta = torch.sum(self.eta * self.eta, dim = (1,2))
        gibbs = (self.get_potential_energy() +
                 self.get_kinetic_energy().squeeze()
                 - torch.sum(self.externalstress[0:3]) * linalg.det(self.h) / 3.0)
        if self.ttime is not None:
            gibbs += (1.5 * n * self.temperature *
                      (self.ttime * self.zeta.squeeze())**2 +
                      3 * self.temperature * (n - 1) * self.zeta_integrated.squeeze())
        else:
            assert self.zeta == 0.0
        if self.pfactor_given is not None:
            gibbs += 0.5 / self.pfact * contractedeta
        else:
            assert contractedeta == 0.0
        return gibbs


    def get_center_of_mass_momentum(self):
        "Get the center of mass momentum."
        return (self.masses * self.velocities).sum(1)

    def zero_center_of_mass_momentum(self, verbose=0):
        "Set the center of mass momentum to zero."
        cm = self.get_center_of_mass_momentum()
        # abscm = torch.sqrt(torch.sum(cm * cm, dim = 1))
        # if verbose and abscm > 1e-4:
        #     self._warning(
        #         self.classname +
        #         ": Setting the center-of-mass momentum to zero "
        #         "(was %.6g %.6g %.6g)" % tuple(cm))
        self.velocities = self.velocities - cm.unsqueeze(1) / self.masses / self._getnatoms()
    

    
    def get_init_data(self):
        "Return the data needed to initialize a new NPT dynamics."
        return {'dt': self.dt,
                'temperature': self.temperature,
                'desiredEkin': self.desiredEkin,
                'externalstress': self.externalstress,
                'mask': self.mask,
                'ttime': self.ttime,
                'tfact': self.tfact,
                'pfactor_given': self.pfactor_given,
                'pfact': self.pfact,
                'frac_traceless': self.frac_traceless}

    def get_data(self):
        "Return data needed to restore the state."
        return {'eta': self.eta,
                'eta_past': self.eta_past,
                'zeta': self.zeta,
                'zeta_past': self.zeta_past,
                'zeta_integrated': self.zeta_integrated,
                'h': self.h,
                'h_past': self.h_past,
                'timeelapsed': self.timeelapsed}

    def _getbox(self):
        "Get the computational box."
        return self.cell

    def _getmasses(self):
        "Get the masses as an 1XNx1 array."
        return self.masses

    def _separatetrace(self, mat):
        """return two matrices, one proportional to the identity
        the other traceless, which sum to the given matrix
        """
        tracePart = ((mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]) / 3.) * torch.eye(3).unsqueeze(0).to(self.device)
        return tracePart, mat - tracePart

    # A number of convenient helper methods
    def _warning(self, text):
        "Emit a warning."
        sys.stderr.write("WARNING: " + text + "\n")
        sys.stderr.flush()

    def _calculate_q_future(self, force):
        "Calculate future q.  Needed in Timestep and Initialization."
        dt = self.dt
        id3 = torch.eye(3).unsqueeze(0).to(self.device)
        alpha = (dt * dt) * torch.bmm(force / self._getmasses(),
                                   self.inv_h)
        beta = dt * torch.bmm(self.h, torch.bmm(self.eta + 0.5 * self.zeta * id3, self.inv_h))
        inv_b = linalg.inv(beta + id3)
        self.q_future = torch.bmm(2 * self.q +
                               torch.bmm(self.q_past, beta - id3) + alpha,
                               inv_b)

    def _calculate_q_past_and_future(self):
        def ekin(p, m=self._getmasses()):
            p2 = torch.sum(p * p, (1,2), keepdim=True)
            return 0.5 * torch.sum(p2 / m, (1,2)) / self._getnatoms()

        
        p0 = self._getmasses() * self.velocities
        m = self._getmasses()
        p = p0.clone().to(self.device)
        dt = self.dt
        for i in range(2):
            self.q_past = self.q - dt * torch.bmm(p / m, self.inv_h)
            self._calculate_q_future(self.get_forces())
            p = torch.bmm(self.q_future - self.q_past, self.h / (2 * dt)) * m
            e = ekin(p)
            if (e < 1e-5).all():
                # The kinetic energy and momenta are virtually zero
                return
            p = (p0 - p) + p0

    def _initialize_eta_h(self):
        self.h_past = self.h - self.dt * torch.bmm(self.h, self.eta)
        if self.pfactor_given is None:
            deltaeta = torch.zeros(6, dtype=torch.float32).to(self.device)
        else:
            deltaeta = ((-self.dt * self.pfact * linalg.det(self.h)).unsqueeze(-1) * (self.get_stress() - self.externalstress))
        if self.frac_traceless == 1:
            self.eta_past = self.eta - self.mask * self._makeuppertriangular(deltaeta)
        else:
            trace_part, traceless_part = self._separatetrace(
                self._makeuppertriangular(deltaeta))
            self.eta_past = (self.eta - trace_part -
                             self.frac_traceless * traceless_part)

    def _makeuppertriangular(self, sixvector):
        "Make an upper triangular matrix from a batch of 6-vectors."
        cell = torch.zeros((sixvector.shape[0], 3, 3), dtype=torch.float32).to(self.device)
        cell[:, 0, 0] = sixvector[:, 0]
        cell[:, 1, 1] = sixvector[:, 1]
        cell[:, 2, 2] = sixvector[:, 2]
        cell[:, 1, 2] = sixvector[:, 3]
        cell[:, 0, 2] = sixvector[:, 4]
        cell[:, 0, 1] = sixvector[:, 5]
        return cell
    
    def _makesixvector(self, mat):
        "Make a 6-vector from a batch of upper triangular matrices."
        return torch.stack([mat[:, 0, 0], mat[:, 1, 1], mat[:, 2, 2],
                        mat[:, 1, 2], mat[:, 0, 2], mat[:, 0, 1]]).T.to(self.device)

    @staticmethod
    def _isuppertriangular(m) -> bool:
        "Check that a matrix is on upper triangular form."
        first = torch.allclose(m[:, 1, 0], torch.zeros_like(m[:, 1, 0]), atol = 1e-6)
        second = torch.allclose(m[:, 2, 0], torch.zeros_like(m[:, 2, 0]), atol = 1e-6)
        third = torch.allclose(m[:, 2, 1], torch.zeros_like(m[:, 2, 1]), atol = 1e-6)
        return first and second and third

    def _calculateconstants(self):
        """(Re)calculate some constants when pfactor,
        ttime or temperature have been changed."""

        n = self._getnatoms()
        if not hasattr(self, "ttime"):
            self.tfact = 0.0
        else:
            self.tfact = 2.0 / (3 * n * self.temperature *
                                self.ttime * self.ttime)
        if not hasattr(self, "pfactor_given"):
            self.pfact = 0.0
        else:
            self.pfact = 1.0 / (self.pfactor_given * linalg.det(self._getbox()))
            # self.pfact = 1.0/(n * self.temperature * self.ptime * self.ptime)
        self.desiredEkin = 1.5 * (n - 1) * self.temperature

    def _setbox_and_positions(self, h, q):
        """Set the computational box and the positions."""
        self.cell = h
        r = torch.bmm(q + 0.5, h)
        self.radii = r

    def _getnreplicas(self):
        """Get the number of replicas."""
        return self.radii.shape[0]
  
    def _getnatoms(self):
        """Get the number of atoms.

        In a parallel simulation, this is the total number of atoms on all
        processors.
        """
        return self.radii.shape[1]

    def _getvolume(self):
        """Get the volume of the system."""
        return linalg.det(self._getbox())

    def _make_special_q_arrays(self):
        """Make the arrays used to store data about the atoms.

        In a parallel simulation, these are migrating arrays.  In a
        serial simulation they are ordinary Numeric arrays.
        """
        self.q = torch.zeros_like(self.radii, float).to(self.device)
        self.q_past = torch.zeros_like(self.radii, float).to(self.device)
        self.q_future = torch.zeros_like(self.radii, float).to(self.device)


class WeakMethodWrapper:
    """A weak reference to a method.

    Create an object storing a weak reference to an instance and
    the name of the method to call.  When called, calls the method.

    Just storing a weak reference to a bound method would not work,
    as the bound method object would go away immediately.
    """

    def __init__(self, obj, method):
        self.obj = weakref.proxy(obj)
        self.method = method

    def __call__(self, *args, **kwargs):
        m = getattr(self.obj, self.method)
        return m(*args, **kwargs)