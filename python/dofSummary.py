#!/usr/bin/env python

# Copyright 2026 Vikram K. Mulligan and Andrew Powers.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# @author: Andrew Powers (apowers4@uoregon.edu)
# @brief: Take the DoF outputs from our model and convert them to
#       peptides in PyRosetta.
#

# import packages
from pyrosetta import init
import pyrosetta.io as io
from pyrosetta.rosetta.core.pose import Pose
import pyrosetta.rosetta.protocols as protocols
import pyrosetta.rosetta.protocols.simple_moves as sm
import pyrosetta.rosetta.protocols.cyclic_peptide as cp
import pyrosetta.rosetta.core.select.residue_selector as select
import pyrosetta.rosetta.core as core
import pyrosetta.rosetta as rosetta
import pyrosetta.rosetta.core.io.silent as silent
from pyrosetta.rosetta.std import map_core_id_AtomID_core_id_AtomID
from pyrosetta.rosetta.core.id import AtomID

from pyrosetta.rosetta.core.scoring import ScoreFunction
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Import numpy
import torch
import os
import warnings
from typing import List

# Import termina bond
from terminal_error import TerminalPeptide


#### Class
class DoFSummary:
    def __init__(
        self,
        ignore_virtualatoms: bool = False,
        prefix: str = 'test',
        extract_minimize_input: bool = False,
        is_cyclic: bool = False,
        pose_motif_location: int = 2,
        ):
        """
        Either extract DoFs from a PDB's pose for input into the model or
        thread DoFs from the model output onto a template pose convert
        DoF vectors to PDB/silentfile

        PARAMS
        ------
        :ignore_virtualatoms: Excludes virtual atoms from DoF extraction/threading
            if passed
        :prefix: Prefix for the outputs of your PDBs/SilentFile. 
        :extract_minimize_input: True/False if True then the entire input peptides
            DoFs are extracted and minimized using CyclicCAE.
        :is_cyclic: True the input PDB binder/peptide is treated as a cyclic peptide
            False it isn't modified.
        :pose_motif_location: The location that our motifs will be placed in our
            ouputted poses.
        """
        self.ignore_virtualatoms = ignore_virtualatoms
        self.prefix = prefix
        self.extract_minimize_input = extract_minimize_input
        self.term_bond = TerminalPeptide()
        self.is_cyclic = is_cyclic
        self.consistent_motif_resi = pose_motif_location
        # Chain string map
        self.chainint2str = dict(
                zip(
                    list(range(1, 27)),
                    "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(",")
                    )
                )
        self.chainstr2int = dict(
                zip(
                    "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(","),
                    list(range(1, 27))
                    )
                )

        # create list of current DOF types in Rosetta
        dof_types = [
            core.id.DOF_Type.PHI,
            core.id.DOF_Type.THETA,
            core.id.DOF_Type.D,
            core.id.DOF_Type.RB1,
            core.id.DOF_Type.RB2,
            core.id.DOF_Type.RB3,
            core.id.DOF_Type.RB4,
            core.id.DOF_Type.RB5,
            core.id.DOF_Type.RB6,
        ]

        # Create a dict that maps int representations to class types
        self.dof_int_to_type = dict(zip([i for i in range(1, len(dof_types)+1)], dof_types))

        # init native peptide, as this is important for switch functions
        self.peptide_native = None

        # init instance of pyrosetta with params
        params = "-mute all -score:weights ref2015.wts -write_all_connect_info -detect_disulf false -ignore_unrecognized_res true -in:auto_setup_metals"
        # params = "-score:weights ref2015.wts -write_all_connect_info"
        init(params)



#### Functions
    def rotate_pose(
        self,
        pose: Pose,
        residue_start: int,
        ) -> Pose:
        """
        Take a pose and place the motif along a consistent residue idx
        This is important for downstream applications where we need to
        specify resi numbers and for consistency in design tasks

        PARAMS
        ------
        :pose: Rosetta Pose that has it's DoFs threaded already and the
            motif residues were added
        :residue_start: The start resi position of our motif (0-index)
            this value comes from the search of what is the most similar
            start resi on our scaffold that matches our motif. This value
            can be different between individual scaffolds in the same run/batch.

        RETURNS
        -------
        :rotated_pose: A pose where motifs are placed at a consistent resi
            specified by `self.consistent_motif_resi`
        """
        # First check if our pose is already what we want
        if self.consistent_motif_resi == residue_start + 1:
            return pose.clone()

        # clone our pose and empty it
        rotated_pose = pose.clone()
        rotated_pose.clear()

        assert True == rotated_pose.empty(), "This pose is not empty and needs to be fixed"

        # Generate a mapping for our scaffold residues to the new desired ordering
        resimap = [ ((ir % pose.total_residue())+1,(new_res % pose.total_residue())+1) for ir,new_res in zip(range(residue_start+1, residue_start+1+pose.total_residue()), range(self.consistent_motif_resi, self.consistent_motif_resi+pose.total_residue())) ]
        sorted_resimap = sorted(resimap, key=lambda x: x[1])

        # Loop through and build our reordered peptide
        for old_scaffold_resi, new_scaffold_resi in sorted_resimap:
            if new_scaffold_resi == 1:
                rotated_pose.append_residue_by_bond(pose.residue(old_scaffold_resi))
            else:
                rotated_pose.append_polymer_residue_after_seqpos(
                    pose.residue(old_scaffold_resi),
                    new_scaffold_resi-1,
                    False,
                )

        return rotated_pose

    def mutate_gly_to_orginial_residue(
        self,
        pose: Pose,
        residue_start: int,
        ) -> int:
        """
        Take residue information stored in `self.motif_residue_info`
        which is precomputed when extracting PDB information in 
        `extract_dof_from_pose()` and mutate motif on Model outputs
        to our desired residues of interset.

        PARAMS
        ------
        :pose: a cloned pose of our poly-glycine structre that has the
            DoFs threaded onto it.
        :residue_start: The start resi position of our motif (0-index)

        RETURNS
        -------
        :pose: Our mutate posed where the residues of interest are now
            restored. While everything else is retained as Glycine.
        """
        def is_in_canonical_twenty( resname ):
            canonical_aa = [
                "ALA","ARG","ASN","ASP",
                "CYS","GLN","GLU","GLY",
                "HIS","ILE","LEU","LYS",
                "MET","PHE","PRO","SER",
                "THR","TRP","TYR","VAL",
                ]
            return resname in canonical_aa
        # Write the phi/psi output of our motif
        outfile_name = "phi_psi_motif.csv"
        if not os.path.exists(outfile_name):
            outfile = open(outfile_name, "w")
#             print("res,phi,psi", file=outfile)
        else:
            outfile = open(outfile_name, "a")
        # iter through our original residue information
        # Mutate GlY -> Original Residue
        for ele, ir in enumerate(list(self.motif_residue_info.keys())):
            mutateResidue = sm.MutateResidue()
            # mutateResidue.preserve_atom_coords()
            residue_number = residue_start.item() + 1 + ele
            if residue_number > pose.total_residue():
                residue_number = residue_number - pose.total_residue()
#             print(str(ir) + "," + str(pose.phi(residue_number)) + "," + str(pose.psi(residue_number)), file=outfile)
            mutateResidue.set_target(residue_number)

            # Determine chirality (This is important when masking atoms, as chirality may flip)
#             if (
#                     (pose.phi(residue_number) > 0) & 
#                     (len(self.motif_residue_info[ir].name().strip()) == 3) &
#                     is_in_canonical_twenty(self.motif_residue_info[ir].name().strip()) &
#                     (self.motif_residue_info[ir].name() != "GLY")
#                     ):
#                 mutateResidue.set_res_name("D"+self.motif_residue_info[ir].name())
#             elif (
#                     (pose.phi(residue_number) < 0) & 
#                     (len(self.motif_residue_info[ir].name().strip()) > 3) & 
#                     is_in_canonical_twenty(self.motif_residue_info[ir].name().strip()[1:])
#                     ):
#                 mutateResidue.set_res_name(self.motif_residue_info[ir].name()[1:])
#             else:
            mutateResidue.set_res_name(self.motif_residue_info[ir].name())
            mutateResidue.apply(pose)

            # Now apply correction to chi angles, so they are more similar
            # to our starting structure
            for ichi in range(1, self.motif_residue_info[ir].nchi()+1):
                pose.set_chi(
                    ichi,
                    residue_number, 
                    self.motif_residue_info[ir].chi(ichi),
                )

        # Fix some output stuff
        self.remove_nme(pose)
        self.remove_all_terminal_types(pose)
        # self.declare_terminal_bond(pose)
        # self.full_cyclize_pose(pose)
        return 0


    def read_pdb_extract_peptide(
        self,
        pdb_path: str,
        residues_of_interest: torch.Tensor,
        peptide_chain: int|str = 2,
        ) -> tuple[Pose, int]:
        """
        Take in a PDB and extract the peptide/binder into
        another pose for DoF Extraction

        PARAMS
        ------
        :pdb_path: Path to the PDB file that containts the peptide/binder
            we want to use for inpainting.
        :residues_of_interest: The residues we want to extract
        :peptide_chain: The integer or char representing the peptide/binders chain
            in our PDB.

        RETURNS
        -------
        :peptide_pose: Our extracted peptide/binder only pose.
        :aa_removed: The amount of amino acids removed (necessar for calculating the relative position of 
                                                        anchor residues)
        """
        # Chain string map
#         chainint2str = dict(zip([1,2,3,4,5,6,7], ["A", "B", "C", "D", "E", "F", "G"]))
        peptide_chain_int = self.chainstr2int[peptide_chain]

        # load in our PDB
        full_pose = io.pose_from_pdb(pdb_path)
        
        #grab length of PDB
        res_len = full_pose.total_residue()

#         # Select the chain we want to extract
#         start_res = full_pose.chain_begin(peptide_chain_int)
#         end_res = full_pose.chain_end(peptide_chain_int)

        # Clone our pose for alignment at the end of the class (when dumping)
        self.full_input = full_pose.clone()

        # clone the pose
#         peptide_pose = full_pose.clone()
        peptide_pose = core.pose.Pose()

        # grab pose numbering for our peptide and Input in general
        pdb_info = full_pose.pdb_info()

        # Problem:
        # Target = 200 Residues
        # Peptide = 8 residues

        # Extract out peptide of interest and load into new pose
        chain_selector = core.select.residue_selector.ChainSelector(peptide_chain)
        subset = chain_selector.apply(full_pose)
        pep_pose_values = [i for i, resi_sel in enumerate(subset, start=1) if resi_sel]
        pep_pdb_values = [int(pdb_info.pose2pdb(i).split()[0]) for i in pep_pose_values]
        pep_pose_vector1 = rosetta.utility.vector1_unsigned_long()
        for resi in pep_pose_values: pep_pose_vector1.append(resi)
        core.pose.pdbslice(peptide_pose, full_pose, pep_pose_vector1)

        # option for if we are extracting all dofs from peptide
        if self.extract_minimize_input:
            residues_of_interest = torch.arange(peptide_pose.total_residue())
#             residues_of_interest = pep_pdb_values

        # Set the fold tree
        if (peptide_pose.total_residue() >= len(residues_of_interest)) and (peptide_pose.total_residue() != 1):
            # Setup our fold tree
            root = self.foldtree_define(peptide_pose.total_residue())
            ft = core.kinematics.FoldTree()
            ft.clear()
            ft.add_edge(1, root, -1)
            ft.add_edge(root, peptide_pose.total_residue(), -1)
            ft.reorder(root)
            peptide_pose.fold_tree(ft)

            updated_residues = list()
            self.pdb2pose_resi_pose = list()
            for ir in residues_of_interest:
                self.pdb2pose_resi_pose.append( pdb_info.pdb2pose(peptide_chain, ir.item()) )
                ir -= pep_pdb_values[0]
                updated_residues.append( ir+1 )
        else:
            assert peptide_pose.total_residue() == len(residues_of_interest), "Our motif & peptide have too many residues (there is a problem with the selection logic for this section as they are not equal)"
            # assert peptide_pose.total_residue() < len(residues_of_interest), "Our peptide has too few residues compared ot our motif (there is a problem with the selection logic for this section as they are not equal)"
            stub_mover = protocols.cyclic_peptide.PeptideStubMover()
            stub_mover.set_reset_mode(False)
            stub_mover.add_residue(
                    stubmode ="Append",
                    resname = "GLY",
                    position = 0,
                    jumpmode = False,
                    connecting_atom = "",
                    repeat = 2,
                    anchor_rsd = peptide_pose.chain_end(1),
                    anchor_rsd_selector = None,
                    anchor_atom = "",
                    )
            stub_mover.add_residue(
                    stubmode ="Prepend",
                    resname = "GLY",
                    position = 0,
                    jumpmode = False,
                    connecting_atom = "",
                    repeat = 2,
                    anchor_rsd = 1,
                    anchor_rsd_selector = None,
                    anchor_atom = "",
                    )
            stub_mover.apply(peptide_pose)
            peptide_pose.update_residue_neighbors()
#             updated_residues = [i+2 for i in range(1, len(residues_of_interest)+1)]
            updated_residues = [i+3 for i in range(len(residues_of_interest))]
            # Set the correct our omega angels to 180.0, since placed as cis.
            for ir in range(1, peptide_pose.size()+1):
                if peptide_pose.residue(ir).is_protein():
                    peptide_pose.set_omega(ir, 180.0)
            # Debug purposes
            peptide_pose.dump_pdb("poly_gly_stub.pdb")
            self.pdb2pose_resi_pose = pep_pose_vector1
        return peptide_pose, torch.tensor(updated_residues)

    def dof_extract(
        self,
        pose: Pose,
        dof_type: int,
        residues_of_interest: torch.Tensor,
        key: torch.Tensor,
        ) -> torch.Tensor:
        """
        Extract a specific type of DoF from the passed pose

        PARAMS
        ------
        :pose: Passed Rosetta Pose object (poly-glycine)
        :dof_type: An integer that represents the dof type we want
            to etract
        :residues_of_interest: Tensor of size (M) which indexes the residues
            we are interested to extract (1-indexed)
        :key: Our specific residue/residues of interest idx for extracting the
            information that we need N-CA-C-O-H

        RETURNS
        -------
        :dof_tensor: A Tensor of size (1, M*A) where the M is the number of residues
            and A is the number of atoms. Where the order of the dofs is N->C
            as in lower residue to higher residue being last.
        """
        # init our tensor as a list since we don't have a determined size yet
        # And store our atom tree
        dof_tensor = list()
        # bond_length_list = list()
        # bond_angle_list = list()
        atom_tree = pose.atom_tree()

        # Loop through our residues of interest and extract DoF info
        if (dof_type == 1) and (not self.extract_minimize_input):
            for ir in residues_of_interest:
                dof_tensor.extend(
                    [
                        pose.phi(ir.item()),
                        pose.psi(ir.item()),
                        pose.omega(ir.item()),
                    ]
                )
        else:
            for ir in range(1, pose.total_residue()+1):
                for ia in range(1, pose.residue(ir).natoms()+1):
                    kinatom = atom_tree.atom( core.id.AtomID(ia, ir) )
                    if not kinatom.is_jump():
                        dof_tensor.append(
                            pose.dof(
                                core.id.DOF_ID(
                                    core.id.AtomID(ia, ir),
                                    self.dof_int_to_type[dof_type],
                                )
                            )
                        )
            # if dof_type == 3:
            #     bond_length_list.extend(torch.tensor(dof_tensor)[key].tolist())
            # elif dof_type == 2:
            #     bond_angle_list.extend(torch.tensor(dof_tensor)[key].tolist())
            # # Clear the dof tensor list
            # dof_tensor.clear()

        # DoF Types 1 and 2 are in radians FYI
        if self.extract_minimize_input and not dof_type == 3:
            dof_tensor = torch.tensor(dof_tensor, dtype=torch.float64)
            dof_tensor = torch.rad2deg(dof_tensor)
        elif self.extract_minimize_input and dof_type == 3:
            dof_tensor = torch.tensor(dof_tensor, dtype=torch.float64)
        else:
            if dof_type == 1:
                dof_tensor = torch.tensor(dof_tensor, dtype=torch.float64)
                dof_tensor = torch.deg2rad(dof_tensor)
#             elif dof_type == 2:
#                 dof_tensor = torch.tensor(dof_tensor, dtype=torch.float64)[key]
#                 dof_tensor = torch.rad2deg(dof_tensor)
            else:
                dof_tensor = torch.tensor(dof_tensor, dtype=torch.float64)[key]

        #     # Add in padding to DoFs
        #     dof_tensor = torch.cat(
        #         (
        #             dof_tensor,
        #             torch.zeros(len(residues_of_interest)*9 - dof_tensor.shape[0]),
        #         ), dim=-1,
        #     )
        #### NOTE: WE NEED TO CALCULATE THE DIHEDRAL OMEGA FOR THE LAST RESIDUE
        #### YOU CAN USE pyrosetta.toolbox.numpy_utils.calc_dihedral(c)
        #### Where c is a np.ndarray of coordinates of four points in space ( atoms)
        return dof_tensor.unsqueeze(0)

    def n_to_c_distance(self, pose: Pose) -> bool:
        """
        Check the distance between the chain, if it is within 2A then count it as
        a cyclic peptide. If not then don't apply cyclic peptide changes to it.

        PARAMS
        -------
        :pose: Rosetta pose object of our peptide/protein

        RETURNS
        -------
        :bool: True it is a cyclic peptide, False it is not
        """
        # Grab the first residue
        n_termini = pose.atom_tree().atom(
            core.id.AtomID(1, 1)
        )
        c_termini = pose.atom_tree().atom(
            core.id.AtomID(3, pose.total_residue())
        )
        distance = n_termini.distance(c_termini)
        if distance > 2:
            return False
        else:
            return True

    def extract_dof_from_pose(
        self,
        pose: Pose,
        residues_of_interest: torch.Tensor,
        ) -> torch.Tensor:
        """
        Take a pose and extract out the desired motifs residues and store them
        in a torch.Tensor array for CycliCAE integration

        PARAMS
        ------
        :pose: Our input starting residues
        :residues_of_interest: Tensor of size (M) which indexes the residues
            we are interested to extract (1-indexed)

        RETURNS
        -------
        :motif_torsion_dofs: a Tensor of size (1, M*A). Where our dihedrals are
            stored from the extracted pose
        :motif_bondangle_dofs: a Tensor of size (1, M*A). Where our dihedrals are
            stored from the extracted pose
        :motif_bondlength_dofs: a Tensor of size (1, M*A). Where our dihedrals are
            stored from the extracted pose
        """
        # cyclic_peptide = self.n_to_c_distance(pose)
        if self.is_cyclic:
            # Make sure there are no strange terminal types
            self.remove_all_terminal_types(pose)
            # Make sure our structure is cyclic
            self.full_cyclize_native(pose)
        else:
            self.remove_all_terminal_types(pose)

        self.peptide_native = pose.clone()

        # If this is True, then we want all of the DoFs and Residue information
        if self.extract_minimize_input:
            residues_of_interest = torch.tensor([i for i in range(1, pose.total_residue()+1)],
                                                dtype=torch.long)
            self.residues_of_interest = residues_of_interest

        # First we extract out the Resiude level information of our motif
        self.motif_residue_info = dict()
        for m in residues_of_interest:
            self.motif_residue_info[m.item()] = pose.residue(m.item())

        # Now mutate our residues to Glycine
        for ir in range(1, pose.total_residue()+1):
            try:
                mutateResidue = sm.MutateResidue()
                mutateResidue.preserve_atom_coords()
                mutateResidue.set_target(ir)
                mutateResidue.set_res_name("GLY")
                mutateResidue.apply(pose)
            except:
                # Extract out important info
                old_res = pose.residue(ir)
                backbone_atoms = ["N", "CA", "C", "O"]
                coords = {atom: old_res.xyz(atom) for atom in backbone_atoms}
                # Create a new GLY residue
                gly_res_type = pose.residue_type_set_for_pose().name_map("GLY")
                gly_residue = core.conformation.ResidueFactory.create_residue(gly_res_type)
                # Manually set the backbone atom positions to the saved coordinates
                for atom in backbone_atoms:
                    gly_residue.set_xyz(atom, coords[atom])
                # Replace the residue without orienting the backbone (we already did it manually)
                pose.replace_residue(ir, gly_residue, orient_backbone=False)
#                 core.conformation.idealize_hydrogens(pose.residue(ir), pose.conformation())
                if ir in self.motif_residue_info.keys():
                    core.conformation.idealize_position(ir, pose.conformation())
# 
        pose.update_residue_neighbors()

        if self.is_cyclic:
            self.full_cyclize_pose_withoutbond(pose)
            self.declare_terminal_bond(pose)
        else:
            self.full_cyclize_pose_withoutbond(pose)

        # grab our dof indexing tensors
        bond_angle_idx, bond_length_idx = self.term_bond.extract_bond_angles(
            pose.total_atoms()-1, # Root atom has no dofs
            L = pose.total_residue(),
            root = pose.fold_tree().root(),
        )

        # DEBUG LINE
#         print(bond_angle_idx.shape, bond_angle_idx.view(-1, 5).shape)
#         print(bond_length_idx.shape, bond_length_idx.view(-1, 5).shape)
        # index our bond angle/length indices by the residues we want
        residues_of_interest_angle_indices = bond_angle_idx.view(-1, 5)[residues_of_interest-1]
        residues_of_interest_length_indices = bond_length_idx.view(-1, 5)[residues_of_interest-1]
#         # DEBUG LINE
#         print(bond_angle_idx.view(-1, 5))
#         print(residues_of_interest_angle_indices.view(-1))

        # Extract DoFs of interest
        motif_torsion_dofs = self.dof_extract(pose, 1, residues_of_interest, None) if self.include_mainchain_torsions else None
        motif_bondangle_dofs = self.dof_extract(pose, 2, residues_of_interest, residues_of_interest_angle_indices.view(-1)) if self.include_mainchain_bondangles else None
        motif_bondlength_dofs = self.dof_extract(pose, 3, residues_of_interest, residues_of_interest_length_indices.view(-1)) if self.include_mainchain_bondlengths else None

        return motif_torsion_dofs, motif_bondangle_dofs, motif_bondlength_dofs

    def extract_all_dofs_from_pdb(
        self,
        input_pdb: str,
        peptide_chain: int,
        include_torsion: bool = True,
        include_bondangle: bool = True,
        include_bondlength: bool = True,
        ) -> torch.Tensor:
        """
        Extract all DoFs from our passed peptide chain in the passed PDB and
        return all DoFs in the correct order CyclicCAE is expecting for encoding

        PARAMS
        ------
        :input_pdb: Path to the pdb file we are etracting information from
        :peptide_chain: The integer representing the peptide/binders chain
            in our PDB.
        :include_torsion: Extract DoF (True/False)
        :include_bondangle: Extract DoF (True/False)
        :include_bondlength: Extract DoF (True/False)

        RETURNS
        -------
        :dofs: All DoFs in a torch.Tensor in the correct order that CyclicCAE is expecting.
        """
        # init important variables
        self.include_mainchain_torsions = include_torsion
        self.include_mainchain_bondangles = include_bondangle
        self.include_mainchain_bondlengths = include_bondlength
        self.ref_chain = peptide_chain

        # extract only the peptide/binder pose
        peptide_pose, updated_res = self.read_pdb_extract_peptide(input_pdb, None, peptide_chain)

        # extract the dofs from our peptide pose
        motif_torsion_dofs, motif_bondangle_dofs, motif_bondlength_dofs = self.extract_dof_from_pose(
            peptide_pose,
            None,
        )

        dofs = torch.cat(
                (
                    motif_torsion_dofs,
                    motif_bondangle_dofs,
                    motif_bondlength_dofs,
                    ), -1
                )

        return dofs, (motif_torsion_dofs, motif_bondangle_dofs, motif_bondlength_dofs)

    def extract_pdb_pose_and_dof(
        self,
        input_pdb: str,
        peptide_chain: int|str,
        residues_of_interest: torch.Tensor,
        include_torsion: bool = True,
        include_bondangle: bool = True,
        include_bondlength: bool = True,
        ) -> torch.Tensor:
        """
        Full function for extracting DoFs from a passed PDB. This will extract Torsion,
        bond angles, and bond lengths by default. Also, this will grab and store our
        residues of interest for future mutation of glycines.

        PARAMS
        ------
        :input_pdb: Path to the pdb file we are etracting information from
        :peptide_chain: The integer representing the peptide/binders chain
            in our PDB.
        :residues_of_interest: Tensor of size (M) which indexes the residues
            we are interested to extract (0-indexed)
        :include_torsion: Extract DoF (True/False)
        :include_bondangle: Extract DoF (True/False)
        :include_bondlength: Extract DoF (True/False)

        RETURNS
        -------
        :motif_torsion_dofs: a Tensor of size (1, M*A). Where our dihedrals are
            stored from the extracted pose
        :motif_bondangle_dofs: a Tensor of size (1, M*A). Where our dihedrals are
            stored from the extracted pose
        :motif_bondlength_dofs: a Tensor of size (1, M*A). Where our dihedrals are
            stored from the extracted pose
        """
        # init important variables
        self.include_mainchain_torsions = include_torsion
        self.include_mainchain_bondangles = include_bondangle
        self.include_mainchain_bondlengths = include_bondlength
        self.ref_chain = peptide_chain

        # extract only the peptide/binder pose
        peptide_pose, updated_res = self.read_pdb_extract_peptide(input_pdb, residues_of_interest, peptide_chain)

        # new test
        self.residues_of_interest = updated_res

        for ir in updated_res:
            assert ir <= peptide_pose.total_residue(), "Residues of interest are too large. The DOFs extracted are from the Chain we are interested in only. Resi %i not found in range (1, %i)" % (ir, peptide_pose.total_residue())

        # extract the dofs from our peptide pose
        motif_torsion_dofs, motif_bondangle_dofs, motif_bondlength_dofs = self.extract_dof_from_pose(
            peptide_pose,
            updated_res,
        )

        return (motif_torsion_dofs, motif_bondangle_dofs, motif_bondlength_dofs), updated_res


    def foldtree_define(self, N: int) -> int:
        """
        Define what the root of our fold tree is

        PARAMS
        ------
        :N (int): Desired size of input structure

        RETURNS
        -------
        :root (int): The root residue index (zero indexed)
        """
        if N % 2 == 0:
                return int(N/2)
        else:
            return int((N-1)/2)

    def remove_nme(self, pose: Pose) -> int:
        """
        Remove NME from the current pose

        PARAMS
        ------
        :pose: pose with NME present
        """
        modvartype = sm.ModifyVariantTypeMover()
        modvartype.set_additional_type_to_remove("N_METHYLATION")
        modvartype.apply(pose)
        pose.update_residue_neighbors()
        return 0

    def declare_terminal_bond(self, pose: Pose) -> int:
        """
        Fix terminal bond

        PARAMS
        ------
        :pose: Pose object
        """
        # Fix termini, though this isn't that important
        declarebond = sm.DeclareBond()
        declarebond.set(
            res1 = pose.total_residue(),
            atom1 = 'C',
            res2 = 1,
            atom2 ='N',
            add_termini = False,
        )
        declarebond.apply(pose)
        pose.update_residue_neighbors()
        return 0

    def remove_all_terminal_types(self, pose: Pose) -> int:
        """
        Removes all terminal types that are on the pose currently

        PARAMS
        ------
        :pose: Pose that has terminal variants
        """
        for ir in range(1, pose.total_residue()+1):
            # Modify the variant types
            modifyvariant_nterm = sm.ModifyVariantTypeMover()
            modifyvariant_nterm.set_additional_type_to_remove("CUTPOINT_UPPER")
            modifyvariant_nterm.set_additional_type_to_remove("LOWER_TERMINUS_VARIANT")
            modifyvariant_nterm.set_additional_type_to_remove("ACETYLATED_NTERMINUS_VARIANT")
            modifyvariant_nterm.set_additional_type_to_remove("ACETYLATED_NTERMINUS_CONNECTION_VARIANT")
#             modifyvariant_nterm.set_residue_selector(select.TrueResidueSelector())
            modifyvariant_nterm.set_residue_selector(select.ResidueIndexSelector(ir))
            modifyvariant_nterm.apply(pose)

            modifyvariant_cterm = sm.ModifyVariantTypeMover()
            modifyvariant_cterm.set_additional_type_to_remove("CUTPOINT_LOWER")
            modifyvariant_cterm.set_additional_type_to_remove("UPPER_TERMINUS_VARIANT")
            modifyvariant_cterm.set_additional_type_to_remove("DIMETHYLATED_CTERMINUS_VARIANT")
            modifyvariant_cterm.set_additional_type_to_remove("CTERM_AMIDATION")
#             modifyvariant_cterm.set_residue_selector(select.TrueResidueSelector())
            modifyvariant_cterm.set_residue_selector(select.ResidueIndexSelector(ir))
            modifyvariant_cterm.apply(pose)
        return 0


    def full_cyclize_native(self, pose: Pose) -> int:
        """
        Cyclize the native PDB pose using a Rosetta based method

        PARAMS
        ------
        :pose: Un cyclized pose
        """
        # Modify the variant types
        modifyvariant_nterm = sm.ModifyVariantTypeMover()
        modifyvariant_nterm.set_additional_type_to_remove("CUTPOINT_UPPER")
        modifyvariant_nterm.set_additional_type_to_remove("LOWER_TERMINUS_VARIANT")
        modifyvariant_nterm.set_additional_type_to_remove("ACETYLATED_NTERMINUS_VARIANT")
        modifyvariant_nterm.set_additional_type_to_remove("ACETYLATED_NTERMINUS_CONNECTION_VARIANT")
        modifyvariant_nterm.set_residue_selector(select.ResidueIndexSelector(1))
        modifyvariant_nterm.apply(pose)

        modifyvariant_cterm = sm.ModifyVariantTypeMover()
        modifyvariant_cterm.set_additional_type_to_remove("CUTPOINT_LOWER")
        modifyvariant_cterm.set_additional_type_to_remove("UPPER_TERMINUS_VARIANT")
        modifyvariant_cterm.set_additional_type_to_remove("DIMETHYLATED_CTERMINUS_VARIANT")
        modifyvariant_cterm.set_residue_selector(select.ResidueIndexSelector(pose.total_residue()))
        modifyvariant_cterm.apply(pose)

        # Fix termini, though this isn't that important
        declarebond = sm.DeclareBond()
        declarebond.set(
            res1 = pose.total_residue(),
            atom1 = 'C',
            res2 = 1,
            atom2 ='N',
            add_termini = False,
        )
        declarebond.apply(pose)
        return 0


    def full_cyclize_pose_withoutbond(self, pose: Pose) -> int:
        """
        Cyclize the pose using a Rosetta based method

        PARAMS
        ------
        :pose: Un cyclized pose
        """
        # Modify the variant types
        modifyvariant_nterm = sm.ModifyVariantTypeMover()
        modifyvariant_nterm.set_additional_type_to_add("CUTPOINT_UPPER")
        modifyvariant_nterm.set_residue_selector(select.ResidueIndexSelector(1))
        modifyvariant_nterm.apply(pose)

        modifyvariant_cterm = sm.ModifyVariantTypeMover()
        modifyvariant_cterm.set_additional_type_to_add("CUTPOINT_LOWER")
        modifyvariant_cterm.set_residue_selector(select.ResidueIndexSelector(pose.total_residue()))
        modifyvariant_cterm.apply(pose)

        # # Fix termini, though this isn't that important
        # declarebond = sm.DeclareBond()
        # declarebond.set(
        #     res1 = pose.total_residue(),
        #     atom1 = 'C',
        #     res2 = 1,
        #     atom2 ='N',
        #     add_termini = True,
        # )
        # declarebond.apply(pose)
        pose.update_residue_neighbors()
        return 0

    def generate_initial_empty_glycine_pose(self, N: int) -> Pose:
        """
        Generate a single cyclic peptide pose of glycines that is of size N

        PARAMS
        ------
        :N (int): Desired size of input structure

        RETURNS
        -------
        :gly_pose (Pose): A Rosetta Pose instance that is all glycines of desired N-mer size
            FoldTree has been updated and is uses the middle residue as a root
            Cutpoints are added to the ends and a peptide bond is declared.
        """
        # fill with glycines
        gly_pose = io.pose_from_sequence(
            seq="G"*N,
            res_type="fa_standard",
            auto_termini=False,
        )

        # Set the fold tree
        root = self.foldtree_define(N)
        ft = core.kinematics.FoldTree()
        ft.clear()
        ft.add_edge(1, root, -1)
        ft.add_edge(root, N, -1)
        ft.reorder(root)
        gly_pose.fold_tree(ft)

        # Cyclize the pose (but don't declare a bond yet)
        self.full_cyclize_pose_withoutbond(gly_pose)
        return gly_pose

    def dof_index_generate(self, N: int, dof_params: torch.Tensor) -> dict:
        """
        Generate a list of DoF_ID

        PARAMS
        ------
        :N: Number of amino acids in our peptide
        :dof_params: A tensor that is of shape (M, 3). Where M is the number 
            of combinations for Atomno * residue number. Columns in the last
            dimension are in the order (Resiude, Atomno, DoF_Type)

        RETURNS
        -------
        :dof_id: A dictionary that pairs our unique identifier for a specific
            atom, residue pair and its dof_type with the Rosetta required
            core.id.DOF_ID placement.
            Key: Residue-Atom-DOF_Type
            Value: core.id.DOF_ID(core.id.AtomID(atomno, resi), core.id.DOF_Type)
        """
        # Dictionary to store our key: Atom-Residue-DOFtype and value: DOF_ID(AtomID, DOF_TYPE) pairs
        dof_id = dict()

        # values in range
        for r in range(dof_params.size(0)):
            # This takes our Dof information and converts it to a DOF_ID
            # Our key is Atom-Residue-DoFtype
            dof_id[str(dof_params[r, 1].item())+"-"+str(dof_params[r, 0].item())+"-"+str(dof_params[r, 2].item())] = core.id.DOF_ID(
                core.id.AtomID(dof_params[r, 1].item(), dof_params[r, 0].item()), # This is (Atomno, Residue)
                self.dof_int_to_type[dof_params[r, 2].item()], # This is DOF_TYPE
            )

        return dof_id

    def minimize_pose_complex_full(
            self, pose: Pose,
            ) -> int:
        """Allow the peptide to minimize, but not along the jump."""
        # Clone our pose
        pose_before = pose.clone()
        pose.pdb_info( core.pose.PDBInfo( pose ) )

        # Set up a movemap
        mm = core.kinematics.MoveMap()
        mm.set_bb(False)
        mm.set_chi(False)
        chainA = select.ChainSelector(1)
        neighborhoodSelector = select.NeighborhoodResidueSelector(chainA, 4.0, True)
        neighborhood_subset = neighborhoodSelector.apply(pose)
        for ir in range(1, pose.total_residue()+1):
            if neighborhood_subset[ir]:
                if pose.residue(ir).chain() == 1:
                    mm.set_bb(ir, True)
                    mm.set_chi(ir, True)
                else:
                    mm.set_chi(ir, True)
            else:
                mm.set_bb(ir, False)
                mm.set_chi(ir, False)

        # Setup cart scorefxn
        scorefxn_cart = self.scorefxn.clone()
        scorefxn_cart.set_weight(core.scoring.cart_bonded, 0.5)
        scorefxn_cart.set_weight(core.scoring.pro_close, 0.0)

        # Set up minimover
        minmover = protocols.minimization_packing.MinMover()
        minmover.movemap(mm)
        minmover.cartesian(True)
        minmover.score_function(scorefxn_cart)
        minmover.min_type("lbfgs_armijo_nonmonotone")
        minmover.max_iter(200)
        minmover.apply(pose)

        self.declare_terminal_bond(pose)

        rmsd = core.scoring.bb_rmsd_including_O(pose_before, pose)
        # print(f"RMSD all heavy atom after minimization: {rmsd:.3f}")
        pose.scores["MinMover_bbheavy_RMSD"] = rmsd
        return 0

    def minimize_pose(
        self,
        pose: Pose,
        ) -> int:
        """
        Apply a minmover to the pose for scoring

        PARAMS
        ------
        :pose: Pose with desired motif residues and the scaffold is poly-glycine

        RETURNS
        -------
        :pose: minimized version of this pose
        """
        # Clone our pose
        pose_before = pose.clone()

        # Set up a movemap
        mm = core.kinematics.MoveMap()
#         mm.set_bb(True)
#         mm.set_chi(True)
        if self.peptide_native != None:
            residue_range = [i for i in range(
                self.consistent_motif_resi,
                self.consistent_motif_resi + len(list(self.motif_residue_info.keys())))]
            for ir in range(1, pose.total_residue()+1):
                if ir in residue_range:
                    mm.set_bb(ir, False)
                    mm.set_chi(ir, False)
                else:
                    mm.set_bb(ir, True)
                    mm.set_chi(ir, True)

        # Setup cart scorefxn
        scorefxn_cart = self.scorefxn.clone()
        scorefxn_cart.set_weight(core.scoring.cart_bonded, 0.5)
        scorefxn_cart.set_weight(core.scoring.pro_close, 0.0)

        # Set up minimover
        minmover = protocols.minimization_packing.MinMover()
        minmover.movemap(mm)
        minmover.cartesian(True)
        minmover.score_function(scorefxn_cart)
        minmover.min_type("lbfgs_armijo_nonmonotone")
        minmover.max_iter(200)
        minmover.apply(pose)

        self.declare_terminal_bond(pose)

        rmsd = core.scoring.bb_rmsd_including_O(pose_before, pose)
        # print(f"RMSD all heavy atom after minimization: {rmsd:.3f}")
        pose.scores["MinMover_bbheavy_RMSD"] = rmsd
        return 0

    def grab_heavybb_atomid_map(
            self,
            pose_reference: core.pose.Pose,
            pose_target: core.pose.Pose,
            residue_anchors: list[int],
            target_start_resi: int = 2,
            full_pose: bool = False,
            ) -> map_core_id_AtomID_core_id_AtomID:
        """Build an AtomID map for superimpose_pose call to function

        PARAMS
        ------
        :pose_reference: The receptor or target we designed macrocycles for. This should have
            the peptide or binder we used as a base/anchor for our peptide generation.
        :pose_target: Our peptide/macrocycle byitself not in the context of the target
        :residue_anchors: A list of residues that we want to align against (1-index)
        :target_start_resi: The starting resi of the targets motif (1-index)
        :full_pose: Bool to compare between if you want to get the full pose RMSD and not just a selection

        RETURNS
        -------
        :core_atomid_map: Our map of AtomIDs for superimpose_pose to function
        """
        # init our map id
        atid_map = map_core_id_AtomID_core_id_AtomID()

        # Setup our backbone atoms
        backbone_atoms = ["N", "CA", "C", "O"]

        # RMSD of the entire backbone heavy atoms, full pose, not specific selection
        if full_pose:
            # loop through our residues
            for ir in range(0, pose_reference.total_residue()):
                # Grab the residues
                fzn_posit = (ir+target_start_resi)
                nat_fzn_posit = ir+residue_anchors[0]
                if fzn_posit > pose_target.total_residue(): fzn_posit = fzn_posit % pose_target.total_residue()
                if nat_fzn_posit > pose_reference.total_residue():  nat_fzn_posit = nat_fzn_posit % pose_reference.total_residue()
                curres = pose_target.residue(fzn_posit)
                refres = pose_reference.residue(nat_fzn_posit)

                # Loop through atoms
                for ia in range(1, curres.natoms()+1):
                    if curres.atom_is_hydrogen( ia ): continue
                    if curres.atom_name( ia ).strip() not in backbone_atoms: continue
                    atid_map[AtomID(ia, fzn_posit)] = AtomID(ia, nat_fzn_posit)

        # Specific Selection RMSD only
        else:
            # loop through our residues
            for ir in range(0, len(residue_anchors)):
                # Grab the residues
                fzn_posit = ir+target_start_resi
                nat_fzn_posit = ir+residue_anchors[0]
                if fzn_posit > pose_target.total_residue(): fzn_posit = fzn_posit % pose_target.total_residue()
                if nat_fzn_posit > pose_reference.total_residue():  nat_fzn_posit = nat_fzn_posit % pose_reference.total_residue()
                curres = pose_target.residue(fzn_posit)
                refres = pose_reference.residue(nat_fzn_posit)

                # Loop through atoms
                for ia in range(1, curres.natoms()+1):
                    if curres.atom_is_hydrogen( ia ): continue
                    if curres.atom_name( ia ).strip() not in backbone_atoms: continue
                    atid_map[AtomID(ia, fzn_posit)] = AtomID(ia, nat_fzn_posit)
        return atid_map

    def grab_atomid_map(
            self,
            pose_reference: core.pose.Pose,
            pose_target: core.pose.Pose,
            residue_anchors: list[int],
            target_start_resi: int = 2,
            ) -> map_core_id_AtomID_core_id_AtomID:
        """Build an AtomID map for superimpose_pose call to function. This
            will be an all-atom alignment, excluding hydrogens.

        PARAMS
        ------
        :pose_reference: The receptor or target we designed macrocycles for. This should have
            the peptide or binder we used as a base/anchor for our peptide generation.
        :pose_target: Our peptide/macrocycle byitself not in the context of the target
        :residue_anchors: A list of residues that we want to align against (1-index)
        :target_start_resi: The starting resi of the targets motif (1-index)

        RETURNS
        -------
        :core_atomid_map: Our map of AtomIDs for superimpose_pose to function
        """
        # init our map id
        atid_map = map_core_id_AtomID_core_id_AtomID()

        # loop through our residues
        for ir in range(0, len(residue_anchors)):
            # Grab the residues
            fzn_posit = ir+target_start_resi 
            nat_fzn_posit = ir+residue_anchors[0]
            if fzn_posit > pose_target.total_residue(): fzn_posit = fzn_posit % pose_target.total_residue()
            if nat_fzn_posit > pose_reference.total_residue(): nat_fzn_posit = nat_fzn_posit % pose_reference.total_residue()
            curres = pose_target.residue(fzn_posit)
            refres = pose_reference.residue(nat_fzn_posit)

            # Assert
            # assert curres.natoms() == refres.natoms(), "The number of atoms do not match"

            # Loop through atoms
            for ia in range(1, curres.natoms()+1):
                if curres.atom_is_hydrogen( ia ): continue
#                 if curres.atom_name( ia ).strip() not in ["N", "CA", "O", "C", "CB"]: continue
                atid_map[AtomID(ia, fzn_posit)] = AtomID(ia, nat_fzn_posit)
        return atid_map

    def calculate_rmsd_between_native(self, curpose: Pose, native_pose: Pose) -> int:
        """
        Measure RMSD against native and current pose

        PARAMS
        ------
        :curpose: Threaded pose that was mutated
        :native_pose: PDB native peptide mutated
        """
        # Now if it is the same size and is cyclic then we want to compare our backbones against the native
        if (native_pose.total_residue() == curpose.total_residue() and self.is_cyclic):
            # Grab the bbheavy atom of our full pose
            bbheavy_atom_map_fullpose = self.grab_heavybb_atomid_map(
                    native_pose,
                    curpose,
                    self.residues_of_interest,
                    self.consistent_motif_resi,
                    True,
                    )
            full_pose_heavybbrmsd = core.scoring.superimpose_pose(
                    curpose,
                    native_pose,
                    bbheavy_atom_map_fullpose,
                    )
#             curpose.update_residue_neighbors()
            curpose.scores["fullpose_bbHeavy_RMSD"] = full_pose_heavybbrmsd

        # Grab our atom id maps for RMS scoring (motif fullatom)
        atom_map_motif = self.grab_atomid_map(
                native_pose,
                curpose,
                self.residues_of_interest,
                self.consistent_motif_resi,
                )
        # Grab bb heavy atoms of motif
        bbheavy_atom_map_motif = self.grab_heavybb_atomid_map(
                native_pose,
                curpose,
                self.residues_of_interest,
                self.consistent_motif_resi,
                )
        # Calculate bbheavy rmsd only
        motif_specific_heavybbrmsd = core.scoring.superimpose_pose(
                curpose,
                native_pose,
                bbheavy_atom_map_motif,
                )
        # Calculate full heavy rmsd of motif
        motif_specific_fullrmsd = core.scoring.superimpose_pose(
                curpose,
                native_pose,
                atom_map_motif,
                )

#       # Add scores to our poses
        curpose.scores["motif_allatom_nothydrogens_RMSD"] = motif_specific_fullrmsd
        curpose.scores["motif_bbHeavy_RMSD"] = motif_specific_heavybbrmsd
        return 0

    def remove_term_variants(self,
                            pose: core.pose.Pose,
                            upper_idx: int,
                            lower_idx: int,
                            ) -> int:
        """
        Remove terminal variants

        PARAMS
        ------
        :pose: Un cyclized pose
        :upper_idx: The (1-index) residue index for the N-termini
        :lower_idx: The (1-index) residue index for the C-termini
        """
        # Modify the variant types
        modifyvariant_nterm = sm.ModifyVariantTypeMover()
        modifyvariant_nterm.set_additional_type_to_remove("CUTPOINT_UPPER")
        modifyvariant_nterm.set_additional_type_to_remove("LOWER_TERMINUS_VARIANT")
        modifyvariant_nterm.set_additional_type_to_remove("ACETYLATED_NTERMINUS_VARIANT")
        modifyvariant_nterm.set_additional_type_to_remove("ACETYLATED_NTERMINUS_CONNECTION_VARIANT")
        modifyvariant_nterm.set_residue_selector(select.ResidueIndexSelector(upper_idx))
        modifyvariant_nterm.apply(pose)

        modifyvariant_cterm = sm.ModifyVariantTypeMover()
        modifyvariant_cterm.set_additional_type_to_remove("CUTPOINT_LOWER")
        modifyvariant_cterm.set_additional_type_to_remove("UPPER_TERMINUS_VARIANT")
        modifyvariant_cterm.set_additional_type_to_remove("DIMETHYLATED_CTERMINUS_VARIANT")
        modifyvariant_cterm.set_additional_type_to_remove("CTERM_AMIDATION")
        modifyvariant_cterm.set_residue_selector(select.ResidueIndexSelector(lower_idx))
        modifyvariant_cterm.apply(pose)
        return 0

    def place_peptide_incontext(
            self,
            pose_reference: core.pose.Pose,
            pose_target: core.pose.Pose,
            ref_chain: int|str,
            residue_anchors: List[int],
            target_start_resi: int = 2,
            ) -> core.pose.Pose:
        """Place a peptide that is by itself in context with the target receptor.
            This will be done along the motif (residue_anchor)

        PARAMS
        ------
        :pose_reference: The receptor or target we designed macrocycles for. This should have
            the peptide or binder we used as a base/anchor for our peptide generation.
        :pose_target: Our peptide/macrocycle byitself not in the context of the target
        :ref_chain: I am going to write this as if there are only 2 chains present, this is 
                the chain to remove.
        :residue_anchors: A list of residues that we want to align against (1-index)
        :target_start_resi: The starting resi of the targets motif (1-index)

        RETURNS
        -------
        :target_peptide: A new pose where our receptor and macrocycle/peptide are now
            placed together.
        """
        # We want to make sure that our ref pose doesnt have any weird term variant types
        chain_pdb2pose = pose_reference.residue( residue_anchors[0] ).chain()
#         for ir in range(pose_reference.chain_begin(self.chainstr2int[ref_chain]), pose_reference.chain_end(self.chainstr2int[self.ref_chain])+1): 
        for ir in range(pose_reference.chain_begin(chain_pdb2pose), pose_reference.chain_end(chain_pdb2pose)+1): 
            self.remove_term_variants(pose_reference, ir, ir)

        # grab pose numbering for our peptide
        pdb_info = pose_reference.pdb_info()
        updated_residues = list()

        # Set the fold tree
        for ir in residue_anchors:
#             if isinstance(ref_chain, str):
#                 updated_residues.append(pdb_info.pose2pdb( self.ref_chain, ir ))
#                 print(self.ref_chain, ir, pdb_info.pose2pdb( self.ref_chain, ir ))
#             else:
#                 updated_residues.append(pdb_info.pose2pdb( self.chainint2str[self.ref_chain], ir ))
            updated_residues.append( ir )

        # generate an atom map 
#         atom_map = self.grab_atomid_map(
        atom_map = self.grab_heavybb_atomid_map(
                pose_reference,
                pose_target,
                updated_residues,
                target_start_resi,
                )

        # Superimpose
        core.scoring.superimpose_pose(
                pose_target,
                pose_reference,
                atom_map,
                )

        # Now delete our ref chain
        ref_pose = pose_reference.clone()
        chain_pdb2pose = ref_pose.residue( residue_anchors[0] ).chain()

        chain_remover = protocols.grafting.simple_movers.DeleteRegionMover(
                ref_pose.chain_begin(chain_pdb2pose),
                ref_pose.chain_end(chain_pdb2pose),
                )
        chain_remover.set_rechain(True)
        chain_remover.apply(ref_pose)
#         ref_pose.delete_residue_range_slow(
#                 ref_pose.chain_begin(chain_pdb2pose),
#                 ref_pose.chain_end(chain_pdb2pose),
#                 )

        # Now combine the two poses
#         core.pose.append_pose_to_pose(
#                 pose_target,
#                 ref_pose,
#                 new_chain = True,
#                 )
        pose_target.append_pose_by_jump(
                ref_pose, 1)
#         core.pose.append_pose_to_pose(
#                 pose_target,
#                 ref_pose,
#                 new_chain = True,
#                 )

        pose_target.update_residue_neighbors()


        core.pose.renumber_pdbinfo_based_on_conf_chains(pose_target)
        # This added because disulfides werent detected and found here:
        # https://forum.rosettacommons.org/node/3932
        pose_target.conformation().detect_disulfides()
        return pose_target.clone()

    def determine_internal_bb_hbonds(
        self,
        currpose: Pose,
        cursel: ResidueSelector,
        scorefxn: ScoreFunction,
        filtername: str = "design_bb_hbonds",
        ) -> int:
        """Count the number of bb internal 

        PARAMS
        ------
        :currpose: The current pose
        :cursel: The selection of the pose
        :scorefxn: The scorefunction used to determine hbonds
        :filtername: The name of our filter that will show up in the score file

        RETURNS
        -------
        Number of internal bb hbonds
        """
        # setup the bb internal hbonds
        bb_hbond_filter = cp.PeptideInternalHbondsFilter()
        # Only set bb hbonds
        bb_hbond_filter.set_hbond_types(
                backbone_backbone_setting = True,
                backbone_sidechain_setting = False,
                sidechain_sidechain_setting = False,
                )
        # Set the scorefxn
        bb_hbond_filter.set_scorefxn(scorefxn)
        # Set selection
        bb_hbond_filter.set_residue_selector(cursel)
        bb_hbond_filter.apply(currpose)

        currpose.scores[filtername] = bb_hbond_filter.report_sm(currpose)
        return bb_hbond_filter.report_sm(currpose)

    def shapeComp(
        self,
        currpose: core.pose.Pose,
        jump_selection: int,
        filtername: str = "shapeComp",
        ) -> float:
        """Calculate the shape complementarity between a jump

        PARAMS
        ------
        :currpose: The filled current pose
        :jump_selection: The jump you want to check between

        RETURNS
        -------
        ShapeComp metric
        """
        # Setup a jump selector
        jump_sel = core.select.jump_selector.JumpIndexSelector()
        jump_sel.jump(jump_selection)
        # init our shape comp
        sc = protocols.simple_filters.ShapeComplementarityFilter()
        sc.set_jump_selector(jump_sel)
        sc.filtered_sc(0.5)
        sc.write_int_area(True)
        sc_score = sc.score(currpose)
        currpose.scores[filtername] = sc_score
        return sc_score

    def calculate_metrics(
            self,
            pose: Pose,
            pepSize: int,
            ) -> int:
        """Compute filtering metrics on backbones generated for quicker
        and easier design selection of peptides.

        PARAMS
        ------
        :pose: Complex pose where the Peptide (A) and Target (B) are
            placed together in a pose.
        :pepSize: The size of the peptide (number of Amino Acids)
        """
        chainASelector = select.ChainSelector(1)
        self.determine_internal_bb_hbonds(
                pose,
                chainASelector,
                scorefxn = self.scorefxn_decompose,
                )
        self.shapeComp(
                pose,
                1,
                )
        pose.scores["PeptideSize"] = pepSize

        return 0

    def apply_dof_to_pose(
        self, 
        pose: Pose, 
        scaffold: int,
        score_outputs: bool,
        random_outputs: bool,
        minimize_pose: bool,
        residue_start: int,
        complex_out: bool,
        ) -> Pose:
        """
        Thread a singular DoF vector onto a pose

        PARAMS
        ------
        :pose: Our poly-gly starting pose
        :scaffold: The current sample we are threading
        :score_outputs: determine if we want to score our pose or not
        :random_outputs: True/False if True then residues don't need to be mutated
            or placed back in a Target context.
        :minimize_pose: Use MinMover to minimize the pose slightly in Rosetta's
            ref2015 scorefunction
        :residue_start: A tensor of size (N, 2) where the residue_start[:, 1]
            is the start location of our residue (0-index)
        :complex_out: Bool determing whether the output, should be outputted with
            the target supplied at the beginning.

        RETURNS
        -------
        :pose: Our threaded pose with all required DoFs.
        """
        # Element along the DoFs of interest this updates to be accurate
        ele = 0
        # store atom tree for updating
        atom_tree = pose.atom_tree()
        for ir in range(1, pose.total_residue()+1):
            for ia in range(1, pose.residue_type(ir).natoms()+1):
                # AtomID of residue ir and atom ia
                kinatom = atom_tree.atom( core.id.AtomID(ia, ir) )
                # If sidechain supply specific sidechain information
                if pose.residue_type(ir).atom_is_sidechain(ia):
                    print("This functionality is still underconstruction, as sidechains \
                    were not implemented in the first pass of this model")
                # Backbone atoms
                elif (not pose.residue_type(ir).atom_is_sidechain(ia)) and (not kinatom.is_jump()):
                    if self.include_mainchain_torsions:
                        pose.set_dof(
                            # Our key is Atom-Residue-DoFtype
                            self.dof_index[str(ia)+"-"+str(ir)+"-1"],
                            torch.deg2rad(self.torsion_dof[scaffold, ele]).item(),
                        )
                    if self.include_mainchain_bondangles:
                        pose.set_dof(
                            # Our key is Atom-Residue-DoFtype
                            self.dof_index[str(ia)+"-"+str(ir)+"-2"],
                            torch.deg2rad(self.bondangle_dof[scaffold, ele]).item(),
                        )
                    if self.include_mainchain_bondlengths:
                        pose.set_dof(
                            # Our key is Atom-Residue-DoFtype
                            self.dof_index[str(ia)+"-"+str(ir)+"-3"],
                            self.bondlength_dof[scaffold, ele].item(),
                        )
                # Done with this atom. Update to the next element within our dof vector
                if not kinatom.is_jump():
                    ele+= 1 


        # Update our pose after all DoFs are placed
        pose.update_residue_neighbors()
        
        # Take size for metrics
        pepSize = pose.size()

        if self.extract_minimize_input:
            self.full_cyclize_pose_withoutbond(pose)
            self.declare_terminal_bond(pose)
            if self.peptide_native != None:
                self.calculate_rmsd_between_native(pose, self.peptide_native)
        # Mutate Residues back to their original types if applicable
        elif not random_outputs:
            try:
                self.mutate_gly_to_orginial_residue(pose, residue_start)
            except:
                print("Residue type is Unknown in Rosetta, so you will get only a Glycine backbone in the output...")
            pose = self.rotate_pose(pose, residue_start)
            self.full_cyclize_pose_withoutbond(pose)
            self.declare_terminal_bond(pose)
            if complex_out:
                pose = self.place_peptide_incontext(
                    pose_reference = self.full_input,
                    pose_target = pose,
                    ref_chain = self.ref_chain,
                    residue_anchors = self.pdb2pose_resi_pose, #self.residues_of_interest,
                    target_start_resi = self.consistent_motif_resi,
                )
        else:
            self.full_cyclize_pose_withoutbond(pose)
            self.declare_terminal_bond(pose)

        if minimize_pose and complex_out:
            self.minimize_pose_complex_full(pose)
            # Calculate internal hbonds
            hbond_set = pose.get_hbonds(exclude_sc=True, exclude_scb=True, exclude_bsc=True)
            pose.scores["bb_hbonds"] = hbond_set.nhbonds()
        elif minimize_pose:
            self.minimize_pose(pose)
            if (self.peptide_native != None) and (not self.extract_minimize_input):
                self.calculate_rmsd_between_native(pose, self.peptide_native)
            # Calculate internal hbonds
            hbond_set = pose.get_hbonds(exclude_sc=True, exclude_scb=True, exclude_bsc=True)
            pose.scores["bb_hbonds"] = hbond_set.nhbonds()
        else:
            if (self.peptide_native != None) and (not self.extract_minimize_input):
                self.calculate_rmsd_between_native(pose, self.peptide_native)
            # Calculate internal hbonds
            hbond_set = pose.get_hbonds(exclude_sc=True, exclude_scb=True, exclude_bsc=True)
            pose.scores["bb_hbonds"] = hbond_set.nhbonds()

        if score_outputs:
            self.scorefxn(pose)
            if complex_out:
                self.calculate_metrics(pose, pepSize) # new feature to calculate metrics
        return pose

    def thread_dofs_to_poses(
        self,
        dof_dict: dict,
        silentfile: bool = False,
        score_outputs: bool = False,
        output_path: str = "./",
        random_outputs: bool = False,
        minimize_pose: bool = True,
        residue_start: torch.Tensor = None,
        complex_out: bool = False,
        ) -> None:
        """
        Take a DoF vector output from CyclicCAE and apply each row to 
        a pose to generate PDB outputs.

        PARAMS
        ------
        :dof_dict: A dict that houses important info such as
            x (Torch.Tensor): Tensor of shape (B, D) where B is the number of 
                output samples, and D is the number of Degrees of freedom.
            params (dict) : A dictionary which houses all other important information
                for declaring DoF metadata which is needed.
        :silentfile: True/False if False then output is written to PDBs by default
        :score_outputs: True/False use Rosetta to score using ref2015 and no relax.
        :output_path: Path to the output directory
        :random_outputs: True/False if True then residues don't need to be mutated
            or placed back in a Target context.
        :minimize_pose: Use MinMover to minimize the pose slightly in Rosetta's
            ref2015 scorefunction
        """
        # Add Output that you are adding to a File
        if os.path.isfile(os.path.join(output_path, f'{self.prefix}.silent')):
            warnings.warn(f"You are appending onto {self.prefix}.silent, as it is already in {output_path}")

        # set a scorefunction
        if score_outputs or minimize_pose:
            self.scorefxn = core.scoring.get_score_function(is_fullatom = True)
#             self.scorefxn.set_weight(core.scoring.hbond_lr_bb, 11.7)
#             self.scorefxn.set_weight(core.scoring.hbond_sr_bb, 11.7)
            self.scorefxn.set_weight(core.scoring.hbond_lr_bb, 10.0)
            self.scorefxn.set_weight(core.scoring.hbond_sr_bb, 10.0)
            self.scorefxn.set_weight(core.scoring.coordinate_constraint, 1.0)
            self.scorefxn.set_weight(core.scoring.atom_pair_constraint, 1.0)
            self.scorefxn.set_weight(core.scoring.dihedral_constraint, 1.0)
            self.scorefxn.set_weight(core.scoring.angle_constraint, 1.0)
            self.scorefxn.set_weight(core.scoring.chainbreak, 10.0)
#             self.scorefxn.set_weight(core.scoring.chainbreak, 1.0)

            self.scorefxn_decompose = self.scorefxn.clone()
            emopts = core.scoring.methods.EnergyMethodOptions(self.scorefxn_decompose.energy_method_options())
            emopts.hbond_options().decompose_bb_hb_into_pair_energies(True)
            self.scorefxn_decompose.set_energy_method_options(emopts)

        x = dof_dict["x"]
        params = dof_dict["params"]
        dof_params = params['dof_residue_metadata'].to(torch.int8)

        # Extract out separate dof vectors per DoF Type
        self.torsion_dof = x[:, :params["torsion_main_n"]]
        self.bondangle_dof = x[:, params["torsion_main_n"]:params["torsion_main_n"]+params["bond_angle_main_n"]]
        self.bondlength_dof = x[:, params["torsion_main_n"]+params["bond_angle_main_n"]:]

        # Determine what to include in DoF Output
        self.include_mainchain_torsions = True if params["torsion_main_n"] > 0 else False
        self.include_mainchain_bondangles = True if params["bond_angle_main_n"] > 0 else False
        self.include_mainchain_bondlengths = True if params["bond_length_main_n"] > 0 else False
        self.include_sidechain_torsions = True if params["torsion_side_n"] > 0 else False
        self.include_sidechain_bondangles = True if params["bond_angle_side_n"] > 0 else False
        self.include_sidechain_bondlengths = True if params["bond_length_side_n"] > 0 else False

        # Generate our dof_ID index
        self.dof_index = self.dof_index_generate(params["protein_len"], dof_params)

        # Generate a pose object 
        pose = self.generate_initial_empty_glycine_pose(params["protein_len"])

        if self.ignore_virtualatoms:
            v1 = 7 # First N virtual atom is the 8th atom (0-index)
            v2_3 = -2 # Last two elements in each dof vector
            self.torsion_dof = torch.cat(
                (
                    sefl.torsion_dof[:, :v1],
                    sefl.torsion_dof[:, v1+1:v2_3],
                ), dim=-1
            )
            self.bondangle_dof = torch.cat(
                (
                    self.bondangle_dof[:, :v1],
                    self.bondangle_dof[:, v1+1:v2_3],
                ), dim=-1
            )
            self.bondlength_dof = torch.cat(
                (
                    self.bondlength_dof[:, :v1],
                    self.bondlength_dof[:, v1+1:v2_3],
                ), dim=-1
            )

        if silentfile:
            opts = silent.SilentFileOptions()
            opts.in_fullatom(True)
            opts.set_binary_output(True)
            silentFile = silent.SilentFileData(opts)

        for s in range(x.size(0)):
            # thread dofs for our sample (s)
            updated_pose = self.apply_dof_to_pose(
                pose=pose.clone(), 
                scaffold = s,
                score_outputs = score_outputs,
                random_outputs = random_outputs,
                minimize_pose = minimize_pose,
                residue_start = residue_start[s] if not random_outputs and not self.extract_minimize_input else None,
                complex_out = complex_out,
            )
            if not silentfile:
                updated_pose.dump_pdb(
                    os.path.join(output_path, f"{self.prefix}_{str(s+1).zfill(6)}.pdb")
                )
            else:
                # Write our struct to a silent file
                silentStruct = silent.BinarySilentStruct(
                    opts, updated_pose, f"{self.prefix}_{str(s+1).zfill(6)}",
                )
                silentFile.add_structure(silentStruct)
            # I am guessing this clears up memory and makes it easier to run
            updated_pose.clear()
        if silentfile:
            # Write all information to silentFile
            silentFile.write_all(
                os.path.join(output_path, f'{self.prefix}.silent')
            )

        return 0


def main() -> int:
    """
    Main function for testing our functions and the script as a whole.
    """
    # pull in a dof vector .pt for testing
    data = torch.load("./binary_import/dof_vectors_clusters_firstten2024-04-11torch_tensor.pt", map_location="cpu", weights_only=False)
    
    # enable our class
    dofSummary = DoFSummary(
        extract_minimize_input=True, # This is for testing full output minimization 
    )

    motif_dof = dofSummary.extract_pdb_pose_and_dof(
        input_pdb = "../xml/vikram_valdiation/6xbf-peptide-target-vikram-complex.pdb",
        peptide_chain = 2,
        residues_of_interest = None,
        # residues_of_interest = torch.tensor([4, 7], dtype=torch.int8),
    )

    # # thread our dofs
    # dofSummary.thread_dofs_to_poses(
    #     data,
    #     silentfile=True,
    #     score_outputs=True,
    #     output_path =  "./",
    #     random_outputs = False,
    #     minimize_pose = True,
    # )

    return 0

if __name__ == "__main__":
    main()
