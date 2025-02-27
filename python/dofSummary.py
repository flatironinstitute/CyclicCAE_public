#!/usr/bin/env python
#
# @author: Andrew Powers (apowers4@uoregon.edu)
# @brief: Take the DoF outputs from our model and convert them to
#       peptides in PyRosetta.
#

# import packages
from pyrosetta import init
import pyrosetta.io as io
from pyrosetta.rosetta.core.pose import Pose
import pyrosetta.rosetta.protocols.simple_moves as sm
import pyrosetta.rosetta.protocols.cyclic_peptide as cp
import pyrosetta.rosetta.core.select.residue_selector as select
import pyrosetta.rosetta.core as core
import pyrosetta.rosetta.core.io.silent as silent

# Import numpy
import torch
import os


#### Class
class DoFSummary:
    def __init__(self, ignore_virtualatoms: bool = False, prefix: str = 'test',):
        """
        Either extract DoFs from a PDB's pose for input into the model or
        thread DoFs from the model output onto a template pose convert
        DoF vectors to PDB/silentfile

        PARAMS
        ------
        :ignore_virtualatoms: Excludes virtual atoms from DoF extraction/threading
            if passed
        :prefix: Prefix for the outputs of your PDBs/SilentFile. 
        """
        self.ignore_virtualatoms = ignore_virtualatoms
        self.prefix = prefix


#### Functions
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

        # Modify the variant types
        modifyvariant_nterm = sm.ModifyVariantTypeMover()
        modifyvariant_nterm.set_additional_type_to_add("CUTPOINT_UPPER")
        modifyvariant_nterm.set_residue_selector(select.ResidueIndexSelector(1))
        modifyvariant_nterm.apply(gly_pose)

        modifyvariant_cterm = sm.ModifyVariantTypeMover()
        modifyvariant_cterm.set_additional_type_to_add("CUTPOINT_LOWER")
        modifyvariant_cterm.set_residue_selector(select.ResidueIndexSelector(N))
        modifyvariant_cterm.apply(gly_pose)

        # Fix termini, though this isn't that important
        declarebond = sm.DeclareBond()
        declarebond.set(
            res1 = 1,
            atom1 = 'N',
            res2 = N,
            atom2 ='C',
            add_termini = False,
        )
        declarebond.apply(gly_pose)

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
        dof_int_to_type = dict(zip([i for i in range(1, len(dof_types)+1)], dof_types))

        # Dictionary to store our key: Atom-Residue-DOFtype and value: DOF_ID(AtomID, DOF_TYPE) pairs
        dof_id = dict()

        # values in range
        for r in range(dof_params.size(0)):
            # This takes our Dof information and converts it to a DOF_ID
            # Our key is Atom-Residue-DoFtype
            dof_id[str(dof_params[r, 1].item())+"-"+str(dof_params[r, 0].item())+"-"+str(dof_params[r, 2].item())] = core.id.DOF_ID(
                core.id.AtomID(dof_params[r, 1].item(), dof_params[r, 0].item()), # This is (Atomno, Residue)
                dof_int_to_type[dof_params[r, 2].item()], # This is DOF_TYPE
            )

        return dof_id

    def apply_dof_to_pose(
            self, 
            pose: Pose, 
            scaffold: int,
            score_outputs: bool,
        ) -> Pose:
        """
        Thread a singular DoF vector onto a pose

        PARAMS
        ------
        :pose: Our poly-gly starting pose
        :scaffold: The current sample we are threading
        :score_outputs: determine if we want to score our pose or not

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
        if score_outputs:
            self.scorefxn(pose)
            # print(f"POSE {scaffold+1} Energy:", pose.energies().total_energy())
        return pose

    def thread_dofs_to_poses(self, dof_dict: dict, silentfile: bool = False,
                             score_outputs: bool = False,
                             output_path: str = "./") -> None:
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
        """
        # init instance of pyrosetta with params
        params = "-mute all -score:weights ref2015.wts"
        init(params)

        # set a scorefunction
        if score_outputs:
            self.scorefxn = core.scoring.get_score_function(is_fullatom = True)

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
    data = torch.load("./dof_vectors_clusters_firstten2024-04-11torch_tensor.pt", map_location="cpu", weights_only=False)
    
    # enable our class
    dofSummary = DoFSummary()

    # thread our dofs
    dofSummary.thread_dofs_to_poses(data, silentfile=True, score_outputs=True)

    return 0

if __name__ == "__main__":
    main()
