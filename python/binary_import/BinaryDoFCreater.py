#!/usr/bin/env python
# @author: Andrew Powers (apowers@flatironinstitute.org)
#
#
#

# packages

import torch
import torch.nn as nn

# native packages
import struct
import argparse


##########
## Classes
##########

class DoFExport:
    def __init__(
        self,
        params: dict,
        model_out: torch.Tensor,
        ):
        # Assign params
        self.params = params
        self.model_out = model_out

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

        # bools_values
        bool_values = [1 if x > 0 else 0 for x in values]

        reference_pose = self.params["reference_pose"]
        reference_len = self.params["ref_pose_len"]

        # Construct the struct format string
        binary_format = 'QQ' + '?'*len(bool_values) + 'Q'*len(param_keys) + 'Q'

        binary_format2 = 'd' * self.model_out.numel()
        
        # Create full value information list
        data_to_pack = [filetype_major, filetype_minor] + bool_values + values + [self.model_out.size(0)]
        data_to_pack2 = self.model_out.flatten().tolist()

        # Use struct to pack information
        packed_data1 = struct.pack(binary_format, *data_to_pack)

        packed_skip = struct.pack('Q', reference_len)

        packed_data2 = struct.pack(binary_format2, *data_to_pack2)

        # write to file
        with open("trajetory_out.bin", "wb") as bfile:
            bfile.write(packed_data1)
            bfile.write(packed_skip)
            bfile.write(reference_pose)
            bfile.write(packed_data2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-out-data", type=str, required=True, help="Trajectories from the model, we want converted to Binary.")
    args = p.parse_args()

    # Load in speicifc dataset
    dataset = torch.load("./dof_vectors2024-03-19torch_tensor.pt")

    # Load in passed model outputs
    out = torch.load(args.model_out_data)

    PARAMS = dataset["params"]

    DoFExport(
        params = PARAMS,
        model_out = out,
    )


 
