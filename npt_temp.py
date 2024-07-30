"""Berendsen NPT dynamics class."""
import warnings
from typing import IO, Optional, Union

import numpy as np

from ase import Atoms, units
from ase.md.nvtberendsen import NVTBerendsen


class NPTBerendsen(NVTBerendsen):
    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature: Optional[float] = None,
        *,
        temperature_K: Optional[float] = None,
        pressure: Optional[float] = None,
        pressure_au: Optional[float] = None,
        taut: float = 0.5e3 * units.fs,
        taup: float = 1e3 * units.fs,
        compressibility: Optional[float] = None,
        compressibility_au: Optional[float] = None,
        fixcm: bool = True,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
    ):
        """Berendsen (constant N, P, T) molecular dynamics.

        This dynamics scale the velocities and volumes to maintain a constant
        pressure and temperature.  The shape of the simulation cell is not
        altered, if that is desired use Inhomogenous_NPTBerendsen.

        Parameters:

        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.

        temperature: float
            The desired temperature, in Kelvin.

        temperature_K: float
            Alias for ``temperature``.

        pressure: float (deprecated)
            The desired pressure, in bar (1 bar = 1e5 Pa).  Deprecated,
            use ``pressure_au`` instead.

        pressure_au: float
            The desired pressure, in atomic units (eV/Å^3).

        taut: float
            Time constant for Berendsen temperature coupling in ASE
            time units.  Default: 0.5 ps.

        taup: float
            Time constant for Berendsen pressure coupling.  Default: 1 ps.

        compressibility: float (deprecated)
            The compressibility of the material, in bar-1.  Deprecated,
            use ``compressibility_au`` instead.

        compressibility_au: float
            The compressibility of the material, in atomic units (Å^3/eV).

        fixcm: bool (optional)
            If True, the position and momentum of the center of mass is
            kept unperturbed.  Default: True.

        trajectory: Trajectory object or str (optional)
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        loginterval: int (optional)
            Only write a log line for every *loginterval* time steps.
            Default: 1

        append_trajectory: boolean (optional)
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.


        """

        NVTBerendsen.__init__(self, atoms, timestep, temperature=temperature,
                              temperature_K=temperature_K,
                              taut=taut, fixcm=fixcm, trajectory=trajectory,
                              logfile=logfile, loginterval=loginterval,
                              append_trajectory=append_trajectory)
        self.taup = taup
        self.pressure = self._process_pressure(pressure, pressure_au)
        if compressibility is not None and compressibility_au is not None:
            raise TypeError(
                "Do not give both 'compressibility' and 'compressibility_au'")
        if compressibility is not None:
            # Specified in bar, convert to atomic units
            warnings.warn(FutureWarning(
                "Specify the compressibility in atomic units."))
            self.set_compressibility(
                compressibility_au=compressibility / (1e5 * units.Pascal))
        else:
            self.set_compressibility(compressibility_au=compressibility_au)


    

    def scale_positions_and_cell(self):
        """ Do the Berendsen pressure coupling,
        scale the atom position and the simulation cell."""

        taupscl = self.dt / self.taup
        stress = self.atoms.get_stress(voigt=False, include_ideal_gas=True)
        old_pressure = -stress.trace() / 3
        scl_pressure = (1.0 - taupscl * self.compressibility / 3.0 *
                        (self.pressure - old_pressure))

        cell = self.atoms.get_cell()
        cell = scl_pressure * cell
        self.atoms.set_cell(cell, scale_atoms=True)

    def scale_velocities(self):
        """ Do the NVT Berendsen velocity scaling """
        tautscl = self.dt / self.taut
        old_temperature = self.atoms.get_temperature()

        scl_temperature = np.sqrt(1.0 +
                                  (self.temperature / old_temperature - 1.0) *
                                  tautscl)
        # Limit the velocity scaling to reasonable values
        if scl_temperature > 1.1:
            scl_temperature = 1.1
        if scl_temperature < 0.9:
            scl_temperature = 0.9

        p = self.atoms.get_momenta()
        p = scl_temperature * p
        self.atoms.set_momenta(p)
        return

    def step(self, forces=None):
        """ move one timestep forward using Berenden NPT molecular dynamics."""

        NVTBerendsen.scale_velocities(self)
        self.scale_positions_and_cell()

        # one step velocity verlet
        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces(md=True)

        p = self.atoms.get_momenta()
        p += 0.5 * self.dt * forces

        if self.fix_com:
            # calculate the center of mass
            # momentum and subtract it
            psum = p.sum(axis=0) / float(len(p))
            p = p - psum

        self.atoms.set_positions(
            self.atoms.get_positions() +
            self.dt * p / self.atoms.get_masses()[:, np.newaxis])

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.  For the same reason, we
        # cannot use self.masses in the line above.

        self.atoms.set_momenta(p)
        forces = self.atoms.get_forces(md=True)
        atoms.set_momenta(self.atoms.get_momenta() + 0.5 * self.dt * forces)

        return forces

    def _process_pressure(self, pressure, pressure_au):
        """Handle that pressure can be specified in multiple units.

        For at least a transition period, Berendsen NPT dynamics in ASE can
        have the pressure specified in either bar or atomic units (eV/Å^3).

        Two parameters:

        pressure: None or float
            The original pressure specification in bar.
            A warning is issued if this is not None.

        pressure_au: None or float
            Pressure in ev/Å^3.

        Exactly one of the two pressure parameters must be different from
        None, otherwise an error is issued.

        Return value: Pressure in eV/Å^3.
        """
        if (pressure is not None) + (pressure_au is not None) != 1:
            raise TypeError("Exactly one of the parameters 'pressure',"
                            + " and 'pressure_au' must"
                            + " be given")

        if pressure is not None:
            w = ("The 'pressure' parameter is deprecated, please"
                 + " specify the pressure in atomic units (eV/Å^3)"
                 + " using the 'pressure_au' parameter.")
            warnings.warn(FutureWarning(w))
            return pressure * (1e5 * units.Pascal)
        else:
            return pressure_au
        


