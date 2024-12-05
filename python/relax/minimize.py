#!/usr/bin/env python
#
# @author: Andrew Powers (apowers@flatironinstitute.org or apowers4@uoregon.edu)
# @main: Perforam an amber relaxation after energy minimization to place dihedral and bond lengths
#       into an appropriate orientation.
#

### Import Portion
import os, sys
import io
import glob
import tqdm
import datetime
import time
# import concurrent.futures as futures
import multiprocessing
import argparse
from collections import defaultdict
from functools import partial
import numpy as np

from typing import List, Dict, Union
from absl import logging
import numpy.typing as npt

import pdbfixer
import openmm
import openmm.app as openmm_app
from openmm import unit

### Class
class LocalMinimize:
    def __init__(
        self,
        rounds: int = 300,
        ff_name: str = "amber14-all.xml",
        max_attempts: int = 3,
        stiffness: float = 10.0,
        tolerance: float = 10.0,
        outprefix: str = "out",
        ):
        """Generate an amber forcefield with the appropriate local minimization

        PARAMS
        ------
        :rounds: Number of rounds to perform minimization
        :ff_name: Name of the forcefield to use within openmm
        :max_attempts: Number of minimization attempts to try
        :stiffness: Our harmonic spring constant (kcal/mol A**2)
        :tolerance: Energy goal for L-BFBS iterations (kcal/mol)
        :outprefix: String for out dir.
        """
        # init variables
        self.energy = unit.kilocalories_per_mole
        self.length = unit.angstrom
        self.forcefield = openmm_app.ForceField(ff_name)
        self.max_iterations = rounds
        self.max_attempts = max_attempts
        self.tolerance = tolerance * self.energy
        self.stiffness = stiffness * self.energy / (self.length**2)
        self.outprefix = outprefix


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
        # print("--- Fixing PDB --")
        # load our pdb in
        fixer = pdbfixer.PDBFixer(file)
        fixer.findMissingResidues()
        fixer.removeHeterogens(False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        with io.StringIO() as f:
            openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
            pdb = f.getvalue()
        # print("--- Loading In PDB --")
        pdb = openmm_app.PDBFile(io.StringIO(pdb))
        # pdb = openmm_app.PDBFile(file)

        # load our pdb topoloy and positions into a modeller
        modeller = openmm_app.Modeller(pdb.topology, pdb.positions)


        # We want to make a dictionary to grab our inputs from
        residues_all = defaultdict(list)
        for c in modeller.topology.chains():
            residues_all[c.index].extend([r for r in c.residues()])

        hydrogens_list = list()

        # if we are only working with a monomer then
        if cyclic:
            if len(list(residues_all.keys())) > 1:
                residues = residues_all[1]
            else: residues = residues_all[0]
            hydrogens_list.extend([a for a in residues[0].atoms() if a.name in ["H3", "H2"]])
            hydrogens_list.extend([a for a in residues[-1].atoms() if a.name == "OXT"])
            # print("Modeller: ", modeller.topology)
            # print("--- Deleting Atoms To Form Cyclic Peptide ---")
            # print(hydrogens_list)
            modeller.delete(hydrogens_list)

            # Now we need to get our N and C atoms from the new topology
            residues_all = defaultdict(list)
            for c in modeller.topology.chains():
                residues_all[c.index].extend([r for r in c.residues()])
            if len(list(residues_all.keys())) > 1:
                residues = residues_all[1]
            else: residues = residues_all[0]
            nterm = [i for i in residues[0].atoms() if i.name == "N" ][0]
            cterm = [i for i in residues[-1].atoms() if i.name == "C" ][0]
            # print("Modeller: ", modeller.topology)
            # print("--- Peptide Bond Added N-C ---")
            modeller.topology.addBond(nterm, cterm)
            # print("Modeller: ", modeller.topology)

        # generate our system
        system = self.forcefield.createSystem(
            modeller.topology, constraints=openmm_app.HBonds,
        )

        # Grab our residues of interest again since the Topology has changed
        residues_all = defaultdict(list)
        for c in modeller.topology.chains():
            residues_all[c.index].extend([r for r in c.residues()])

        self._addHarmonicRestraint(
            system,
            modeller,
            residues_all[1] if len(list(residues_all.keys())) > 1 else residues_all[0],
            self.stiffness,
        )

        # Set up our simulation
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName("CUDA" if gpu else "CPU")
        simulation = openmm_app.Simulation(
            modeller.topology, system, integrator, platform,
        )

        # Set our positions
        simulation.context.setPositions(modeller.positions)

        # Set up our simulation run variables, like dictionary and our minimize run
        sim_dict = {}
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        sim_dict["energyInit"] = state.getPotentialEnergy().value_in_unit(self.energy)
        sim_dict["posInit"] = state.getPositions(asNumpy=True).value_in_unit(self.length)

        # Run our simulation to minimize
        simulation.minimizeEnergy(
            # tolerance=self.tolerance,
            maxIterations=self.max_iterations,
        )

        # get our minimized outputs
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        sim_dict["energyFinal"] = state.getPotentialEnergy().value_in_unit(self.energy)
        sim_dict["posFinal"] = state.getPositions(asNumpy=True).value_in_unit(self.length)
        sim_dict["min_pdb"] = self._get_pdb_string(simulation.topology, state.getPositions())

        # extract out file suffix
        fileprefix = file.split('/')[-1].split('.')[0]

        # Make our out dir if not present already
        if not os.path.exists(self.outprefix):
            os.mkdir(self.outprefix)

        # Write our pdb to a file
        with open(os.path.join(self.outprefix, f"{fileprefix}-relaxed.pdb"), "w") as f:
            print(sim_dict["min_pdb"], file=f)


        # print("Initial Energy: ", sim_dict["energyInit"], " --> ", sim_dict["energyFinal"])
        # Calculate the RMSD changes per chains
        rmsdA, rmsdB = self._calculateRMSD(np.array(sim_dict["posInit"]), np.array(sim_dict["posFinal"]), residues_all)

        # Now if our status update file isn't real create
        statusfile = os.path.join(self.outprefix, "out-info.csv")
        if not os.path.exists(statusfile):
            with open(statusfile, "w") as fout:
                print("FILE,DATE,RMSD A, RMSD B,Init Potential Energy,Final Potential Energy", file=fout)
        
        with open(statusfile, "a") as fout:
            print(f"{fileprefix},{datetime.date.today()},{rmsdA},{rmsdB},{sim_dict['energyInit']},{sim_dict['energyFinal']}", file=fout)
        # print(f"RSMD Difference for Chain A: {rmsdA}, Chain B: {rmsdB}")
        return sim_dict

    def _calculateRMSD(
            self,
            pos1: npt.ArrayLike,
            pos2: npt.ArrayLike, 
            topology_dict: Dict[int, list],
        ) -> Union[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Generate an RMSD Difference from ChainA and Peptide for DEBUG and output information

        PARAMS
        ------
        :pos1: Initial array of positions before minimization
        :pos2: Final array of positions after minimization
        :topology_dict: A dictionary of chain index as key and a list of residue objects

        RETURNS
        -------
        :rmsd_A: RMSD difference in angstrom for Target
        :rmsd_B: RMSD difference in angstrom for peptide
        """
        # Determine if there are multiple chains (complex) or a single chain (monomer)
        complex = len(list(topology_dict.keys())) > 1

        # generate our arrays based on size input and if our 
        if complex:
            test = [a for r in topology_dict[1] for a in r.atoms()]
            atoms_A = [a.index for r in topology_dict[0] for a in r.atoms() if a.name in ["N", "CA", "C", "O"]]
            atoms_B = [a.index for r in topology_dict[1] for a in r.atoms() if a.name in ["N", "CA", "C", "O"]]
            # We want a (2, R, 3) Matrix 
            pos_A = np.zeros(
                (
                2,
                len(atoms_A),
                3,
                ),
                dtype=np.float32,
            )
            pos_B = np.zeros(
                (
                2,
                len(atoms_B),
                3,
                ),
                dtype=np.float32,
            )

            # Now place our arrays correctly
            pos_A[0] = pos1[atoms_A]
            pos_B[0] = pos1[atoms_B]
            pos_A[1] = pos2[atoms_A]
            pos_B[1] = pos2[atoms_B]

            # Now perform RMSD
            rmsd_A = np.sqrt(np.sum((pos_A[0] - pos_A[1])**2) / pos_A.shape[1])
            rmsd_B = np.sqrt(np.sum((pos_B[0] - pos_B[1])**2) / pos_B.shape[1])

        else:
            atoms_B = [a.index for r in topology_dict[0] for a in r.atoms() if a.name in ["N", "CA", "C", "O"]]
            # We want a (2, R, 3) Matrix 
            pos_B = np.zeros(
                (
                2,
                len(atoms_B),
                3,
                ),
                dtype=np.float32,
            )

            # Now place our arrays correctly
            pos_B[0] = pos1[atoms_B]
            pos_B[1] = pos2[atoms_B]

            # Now perform RMSD
            rmsd_A = np.array(0)
            rmsd_B = np.sqrt(np.sum((pos_B[0] - pos_B[1])**2) / pos_B.shape[1])

        return rmsd_A, rmsd_B


    def _addHarmonicRestraint(
        self,
        system: openmm.System,
        ref_pdb: openmm_app.Modeller,
        minimize_residues: List[int],
        stiffness: unit.Unit,
        ):
        """Add a harmonic potential restraint to our system

        PARAMS
        ------
        :system: Our current setup system with implict water
        :ref_pdb: Our current topology object
        :minimize_residues: Residues we would like to minimize
        :stiffness: Our spring constant (k) within our force (kcal/mol A**2)
        """
        # Specify our force and add parameters
        custom_harmonic_force = openmm.CustomExternalForce(
            "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        )
        custom_harmonic_force.addGlobalParameter("k", stiffness)
        custom_harmonic_force.addPerParticleParameter("x0")
        custom_harmonic_force.addPerParticleParameter("y0")
        custom_harmonic_force.addPerParticleParameter("z0")

        # Add particles to minimize
        for i, a in enumerate(ref_pdb.topology.atoms()):
            if a.residue not in minimize_residues:
                continue
            elif a.name == "CA":
                custom_harmonic_force.addParticle(i, ref_pdb.positions[i])
        # print("Minimizing %d / %d particles." % (custom_harmonic_force.getNumParticles(), system.getNumParticles()))
        logging.info("Minimizing %d / %d particles.",
                     custom_harmonic_force.getNumParticles(), system.getNumParticles(),
                     )
        # Add our force to the system
        system.addForce(custom_harmonic_force)

    def _get_pdb_string(
        self,
        topology: openmm_app.Topology,
        positions: unit.Quantity,
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
    p.add_argument('--input-file', type=str, default=None, help="Test File To Minimize PDB path File")
    p.add_argument('--input-dir', type=str, default=None, help="Directory of pdb files")
    p.add_argument('--output-dir', type=str, default="out", help="Specify out dir")
    p.add_argument('--max-workers', type=int, default=multiprocessing.cpu_count()-2, help="Number of workers")
    p.add_argument('--gpu', action="store_true", help="Run on GPU")
    args = p.parse_args()

    relax = LocalMinimize(outprefix=args.output_dir)
    if args.input_dir == None:
        relax.minimizeStructure(args.input_file, gpu=args.gpu)
    else:
        file_out = glob.glob(os.path.join(args.input_dir, "fixed_test*.pdb"))
        partial_func = partial(relax.minimizeStructure, gpu=args.gpu)
        # for file in tqdm.tqdm(file_out, total=len(file_out), ncols=100):
            # relax.minimizeStructure(file, gpu=args.gpu)
        print("CPUS-2: ", args.max_workers)
        with multiprocessing.Pool(processes=args.max_workers) as p, tqdm.tqdm(total=len(file_out)) as pbar:
            for result in p.imap(partial_func, file_out):
                pbar.update()
                pbar.refresh()
