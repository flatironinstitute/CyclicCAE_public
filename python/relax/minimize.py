#!/usr/bin/env python
#
# @author: Andrew Powers (apowers@flatironinstitute.org or apowers4@uoregon.edu)
# @main: Perforam an amber relaxation after energy minimization to place dihedral and bond lengths
#       into an appropriate orientation.
#

### Import Portion
import os, sys
import io
import argparse
from collections import defaultdict
from typing import List, Dict
from absl import logging
import numpy as np
import openmm
import openmm.app as openmm_app
import openmm.units as units

### Class
class LocalMinimize:
    def __init__(
        self,
        rounds: int = 3,
        ff_name: str = "amber14-all.xml",
        max_attempts: int = 3,
        stiffness: float = 1.0,
        tolerance: float = 10.0,
        ):
        """Generate an amber forcefield with the appropriate local minimization

        PARAMS
        ------
        :rounds: Number of rounds to perform minimization
        :ff_name: Name of the forcefield to use within openmm
        :max_attempts: Number of minimization attempts to try
        :stiffness: Our harmonic spring constant (kcal/mol A**2)
        :tolerance: Energy goal for L-BFBS iterations (kcal/mol)
        """
        # init variables
        self.energy = units.kilocalories_per_mole
        self.length = units.angstrom
        self.forcefield = openmm_app.forcefield(ff_name)
        self.max_iterations = rounds
        self.max_attempts = max_attempts
        self.tolerance = tolerance * self.energy
        self.stiffness = stiffness * self.energy * (1/self.length**2)


    def minimizeStructure(
        self,
        file: str,
        cyclic: bool = True,
        gpu: bool = False,
        ) -> Dict[str, float]:
        """Use harmonic minimization to minimize a passed peptide

        PARAMS
        ------
        :file: Specify the pdb file to minimize
        :complex: This is a monomer or a cyclic peptide monomer
        :gpu: Use a gpu or not (False/True)
        """
        print("--- Loading In PDB --")
        # load our pdb in
        pdb = openmm_app.PDBFile(file)

        # load our pdb topoloy and positions into a modeller
        modeller = openmm_app.Modeller(pdb.topology, pdb.positions)

        # We want to make a dictionary to grab our inputs from
        residues_all = defaultdict(list)
        for c in modeller.topology.chains():
            residues_all[c.index].extend([r for r in c.residues()])

        # if we are only working with a monomer then
        if cyclic:
            residues = residues_all[1]
            nterm = [i for i in residues[0].atoms() if i.name == "N" ][0]
            cterm = [i for i in residues[-1].atoms() if i.name == "C" ][0]
            modeller.topology.addBond(nterm, cterm)
            print("--- Peptide Bond Added N-C ---")

        # generate our system
        system = self.forcefield.createSystem(
            modeller.topology, constraints=openmm_app.HBonds,
        )
        self._addHarmonicRestraint(
            system,
            pdb,
            residues_all[1],
            self.stiffness,
        )

        # Set up our simulation
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName("CUDA" if gpu else "CPU")
        simulation = openmm_app.Simulation(
            modeller.topology, system, integrator, platform,
        )

        # Set up our simulation run variables, like dictionary and our minimize run
        sim_dict = {}
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        sim_dict["energyInit"] = state.getPotentialEnergy().value_in_unit(units.kilocalories_per_mole)
        sim_dict["posInit"] = state.getPositions(asNumpy=True).value_in_unit(units.angstrom)

        # Run our simulation to minimize
        simulation.minimizeEnergy(
            self.tolerance,
            self.max_iterations,
        )

        # get our minimized outputs
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        sim_dict["energyFinal"] = state.getPotentialEnergy().value_in_unit(units.kilocalories_per_mole)
        sim_dict["posFinal"] = state.getPositions(asNumpy=True).value_in_unit(units.angstrom)
        sim_dict["min_pdb"] = self._get_pdb_string(simulation.topology, simulation.positions)
        return sim_dict

    def _addHarmonicRestraint(
        self,
        system: openmm.System,
        ref_pdb: openmm_app.PDBFile,
        minimize_residues: List[int],
        stiffness: units.Unit,
        ):
        """Add a harmonic potential restraint to our system

        PARAMS
        ------
        :system: Our current setup system with implict water
        :ref_pdb: PDB which was read in and has a topology object
        :minimize_residues: Residues we would like to minimize
        :stiffness: Our spring constant (k) within our force (kcal/mol A**2)
        """
        # Specify our force and add parameters
        custom_harmonic_force = openmm_app.CustomExternalForce(
            "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        )
        custom_harmonic_force.addGlobalParameter("k", stiffness)
        custom_harmonic_force.addPerParticleParameter("x0")
        custom_harmonic_force.addPerParticleParameter("y0")
        custom_harmonic_force.addPerParticleParameter("z0")

        # Add particles to minimize
        for i, a in enumerate(ref_pdb.topology.atoms()):
            if a.residue.index not in minimize_residues:
                continue
            elif a.name == "CA":
                custom_harmonic_force.addParticle(i, ref_pdb.positions[i])
        logging.info("Minimizing %d / %d particles.",
                     custom_harmonic_force.getNumParticles(), system.getNumParticles(),
                     )
        # Add our force to the system
        system.addForce(custom_harmonic_force)

    def _get_pdb_string(
        self,
        topology: openmm_app.Topology,
        positions: units.Quantity,
        ):
        """Get a PDB string from our OpenMM simulation topology and positions

        PARAMS
        ------
        :topology: Simulation topoloyg object
        :positions: Simulation atom positions

        RETURNS
        -------
        :(Str): our written pdb file string
        """
        with io.StringIO() as f:
            openmm_app.PDBFile.writeFile(topology, positions, f)
        return f.getvalue()

    # def _run_minimize(
    #     self,
    #     pdb_input_string: str,
    #     )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input-file', type=str, help="Test File To Minimize PDB path File")
    args = p.parse_args()

    relax = LocalMinimize()
    relax.minimizeStructure(args.input_file)
