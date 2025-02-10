#!/usr/bin/env python
# @author: Andrew Powers (apowers@flatironinstitute.org)
#
#
#

# packages

import torch
import torch.nn as nn
import numpy as np

# native packages
import struct
import argparse
import re


##########
## Classes
##########

class DoFExport:
    def __init__(
        self,
        params: dict,
        model_out: torch.Tensor,
        torsion_only: bool = False,
        no_energy: bool = False,
        peptide_size: int = 8,
        ):
        # Assign params
        self.params = params
        self.model_out = model_out
        self.torsion_only = torsion_only
        self.no_energy = no_energy
        self.peptide_size = peptide_size

        self.extract_to_binary



    
    @property
    def extract_to_binary(self):
        # Explicit version set
        filetype_major = self.params["filetype_major"]
        filetype_minor = self.params["filetype_minor"]

        # Extract out model informatoin
        param_keys = ["torsion_main_n", "torsion_side_n", "bond_angle_main_n", "bond_angle_side_n",
                      "bond_length_main_n", "bond_length_side_n", "jump_n", "energy_n"]

        # Build packing value list
        values = [self.params[k] for k in param_keys]

        if self.torsion_only:
            # change all values to zero, but torsion
            values = [k if n < 2 else 0 for n,k in enumerate(values)]

        if self.no_energy:
            # change all energy and jump to zero
            values = [k if n < 6 else 0 for n,k in enumerate(values)]

        # set up bool values
        bool_values = [1 if x > 0 else 0 for x in values]

        reference_pose = self.params["reference_pose"]
        if self.peptide_size == 8:
            reference_pose = (reference_pose.decode('latin-1')[2968:]).encode('utf-8')
        elif self.peptide_size == 9:
            reference_pose = (reference_pose.decode('latin-1')[516:]).encode('utf-8')

        # reference_len = self.params["ref_pose_len"]

        # Construct the struct format string
        binary_format = 'QQ' + '?'*len(bool_values) + 'Q'*len(param_keys) + 'Q'

        # Extract out the dof residue metadata
        metadata = self.params["dof_residue_metadata"]
        # metadata_size = self.params["torsion_main_n"] if self.torsion_only else metadata.size(0).item()
        if self.torsion_only:
            metadata_size = self.params["torsion_main_n"] 
            print("MET SIZE:", metadata_size)
            dofid_entries = [self.params["torsion_main_n"]]
        elif self.no_energy:
            metadata_size = self.params["torsion_main_n"] + self.params["torsion_side_n"] + self.params["bond_angle_main_n"] + self.params["bond_angle_side_n"] + self.params["bond_length_main_n"] + self.params["bond_length_side_n"]
            dofid_entries = [self.params["torsion_main_n"] + self.params["torsion_side_n"] + self.params["bond_angle_main_n"] + self.params["bond_angle_side_n"] + self.params["bond_length_main_n"] + self.params["bond_length_side_n"]]
        else:
            metadata_size = metadata.size(0).item()
            dofid_entries = [self.params["total_n_dof"]]

        binary_format_residue_metadata = '<'+ 'QQB' * metadata_size

        # print(type(self.model_out))
        # print(self.model_out.keys())
        self.model_out = self.model_out['x'] if type(self.model_out) == dict else self.model_out
        # print(type(self.model_out))
        # print(self.model_out)
        binary_format2 = 'd' * self.model_out.numel()
        
        # Create full value information list
        # Added in DOFid_entries flexible
        # dofid_entries = [self.params["torsion_main_n"]] if self.torsion_only else [self.params["total_n_dof"]]

        data_to_pack = [filetype_major, filetype_minor] + bool_values + values + dofid_entries

        # convert tensor values to, QQB: unint64, uint64, uint8 
        # This is residue index, atom index, dof type
        data_to_pack_residue_metadata = list()
        for row in range(metadata_size):
            for i in range(3):
                if i == 2:
                    ele = metadata[row, i].numpy().astype(np.uint8)
                    data_to_pack_residue_metadata.append(ele)
                else:
                    ele = metadata[row, i].numpy().astype(np.uint64)
                    data_to_pack_residue_metadata.append(ele)

        print(data_to_pack_residue_metadata)
        print(len(data_to_pack_residue_metadata))


        data_to_pack_samplenum = self.model_out.size(0)

        data_to_pack2 = self.model_out.flatten().tolist()

        # Use struct to pack information
        packed_data1 = struct.pack(binary_format, *data_to_pack)

        packed_residue_metadata = struct.pack(binary_format_residue_metadata,
                                              *data_to_pack_residue_metadata)

        packed_samplenum = struct.pack('Q', data_to_pack_samplenum)

        packed_data2 = struct.pack(binary_format2, *data_to_pack2)

        reference_len = len(reference_pose)
        print(reference_len)
        print(reference_pose)

        packed_skip = struct.pack('Q', reference_len)

        # write to file
        with open("trajetory_out.bin", "wb") as bfile:
            bfile.write(packed_data1)
            bfile.write(packed_residue_metadata)
            bfile.write(packed_samplenum)
            bfile.write(packed_skip)
            bfile.write(reference_pose)
            bfile.write(packed_data2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-out-data", type=str, required=True, help="Trajectories from the model, we want converted to Binary.")
    # p.add_argument("--dataset-used", type=str, default="./dof_vectors2024-03-19torch_tensor.pt",
    #                help="Set Dataset Used to Train model (default: dof_vectors2024-03-19torch_tensor.pt)")
    p.add_argument("--datasetloader", type=str, default="../dataload_6to9_01022025.pt", help="Load the dataloader params dictionary pt file")
    p.add_argument("--peptide-size", type=int, default=8, help="Specify the size of the peptide to determine the params to load")
    p.add_argument("--torsion-only", action="store_true", help="Only Send Torsion Data")
    p.add_argument("--no-energy", action="store_true", help="Exclude Jump and Energy Term")
    args = p.parse_args()

    # Load in speicifc dataset
    dataset = torch.load(args.datasetloader)

    # Load in passed model outputs
    out = torch.load(args.model_out_data, map_location="cuda:0" if torch.cuda.is_available() else "cpu")

    PARAMS = dataset[args.peptide_size] #["params"]

    DoFExport(
        params = PARAMS,
        model_out = out,
        torsion_only = args.torsion_only,
        no_energy = args.no_energy,
        peptide_size = args.peptide_size,
    )


 
