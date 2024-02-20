#!/usr/bin/env python
################################################################################
##
## @file python/binary_import/BinaryDoFReader.py
## @brief Allows import of a set of binary degree-of-freedom vectors and an
## RMSD matrix, and conversion to PyTorch input vectors.
## @author Vikram K. Mulligan (vmulligan@flatironinstitute.org), Center for
## Computational Biology, Flatiron Institute.
##
################################################################################


##################################################################################################################
#                                               Packages 
##################################################################################################################

# base python packages
import struct
from collections import defaultdict
import time
import re
import datetime

# import torch
import torch


##################################################################################################################
#                                                Classes
##################################################################################################################

class DoFBinaryReader:
    def __init__(self, binary_file: str, rmsd_binary: str, save_dataset: bool = False, rmsd: bool = True):
        """Custom Binary Rosetta Read In Class

        PARAMS
        ------
            binary_file: str
                Path to binary rosetta/C++ out .dat file

            rmsd_binary: str
                Path to binary rosetta/C++ RMSD calculated .dat file;
                this needs to be set if `rmsd` is True.

            save_dataset: bool
                Defaults to False, but this is used to specify if you would
                like to torch.save the dataset.

            rmsd: bool
                Defaults to True, calculate the RMSD PyTorch Matrix
        """
        # save passed args
        self.binary_file = binary_file
        self.save_dataset = save_dataset
        self.rmsd_binary = rmsd_binary
        self.rmsd = rmsd

        # Set storing header information
        # defaultdict to hold information
        self.metadata_info = defaultdict(int)
        # key values that will be used within the newly init defaultdict above
        self.param_set = ["filetype_major", "filetype_minor", "torsion_main_bool", "torsion_side_bool",
                          "bond_angle_main_bool", "bond_angle_side_bool", "bond_length_main_bool", 
                          "bond_length_side_bool", "jump_bool", "energy_bool", "torsion_main_n", 
                          "torsion_side_n", "bond_angle_main_n", "bond_angle_side_n", "bond_length_main_n",
                          "bond_length_side_n", "jump_n", "energy_n", "nstructs", "bytes_to_skip"] 

        # Format to use for unpacking the binary information
        self.format = [("Q", 8), ("Q", 8), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8)]

        # fill in metadata_info
        print("***EXTRATING HEADER METADATA***")
        self.skip_chunk, self.data_per_sample = self.read_header

        # build extract information data format
        self.data_format, self.num_values = self.binary_data_format

        # Bug check
        assert self.data_per_sample == self.num_values, "The data per sample doesn't match the returned amount"

        # create zeros tensor torch 
        self.data_tensor = torch.zeros(size=(self.metadata_info["nstructs"], self.num_values), dtype=torch.float32)

        # build format given header info for grabbing data
        print("***EXTRACTING BINARY RAW DoF VALUES***")
        with open(self.binary_file, 'rb') as f:
            
            # Skip the amount we need to skip
            header = f.read(self.skip_chunk)

            # grab out the size of the protein
            protein_seq = re.search(r"SEQUENCE:.*?([A-Za-z]+)\n", header.decode('latin-1'))
            self.protein_len = len(protein_seq.group(1))

            # track sample idx
            tensor_sample = 0
            data_idx = 0

            # true for not end
            not_end = True

            # start time 
            dt0 = time.time()

            while not_end:

                # Consistently read in 8 bytes of data, since everything is float64
                binary_data = f.read(8)

                # check if the read in informatin is empty 
                if (not binary_data): 

                    # print out infromation on what sample number the process is ending on
                    print(f"PROCESSING ENDS ON SAMPLE {tensor_sample}!!!!!! TOTAL TIME: {round(time.time() - dt0, 0)}s")

                    # break the while loop
                    not_end = False
                    break

                # unpack information
                self.data_tensor[tensor_sample, data_idx] = struct.unpack('d', binary_data)[0]

                # increment data value idx
                data_idx += 1

                # if the correct number of data values to fill the second D dimension is filled then we will call that sample complete
                if data_idx == self.num_values:

                    # Visual Update on binary read in process
                    if tensor_sample % 10000 == 0:
                        print(f"***Sample {tensor_sample} PROCESSED OUT OF {self.metadata_info["nstructs"]} SAMPLES. ELAPSED TIME {round(time.time() - dt0, 0)}***")

                    # increment sample idx and reset data idx
                    tensor_sample += 1
                    data_idx = 0


        # extract rmsd information if able
        if self.rmsd:
            self.rmsd_tensor = self.rmsd_calc

            


    #################################################################################################################
    #                                        Class Functions
    #################################################################################################################

    @property
    def extract_tensor(self):
        """return and/or save the tensor to disk for future use"""
        
        # If the tensor is marked for saving then go through the process of saving
        if self.save_dataset:

            # extract out params that are useful for the model
            n_param_dict = {k:v for k,v in self.metadata_info.items() if k.endswith("_n")}
            n_param_dict["output_size"] = self.num_values
            n_param_dict["input_size"]  = self.num_values
            n_param_dict["normalize"]   = True
            n_param_dict["protein_len"] = self.protein_len
            
            # Generate the output file name
            out_file_pre = self.binary_file.split("/")[-1].split(".")[0]
            out_file = out_file_pre + str(datetime.date.today()) + "torch_tensor.pt"

            if self.rmsd:
                # save to a tensor dict
                torch.save(
                    {
                        "x": self.data_tensor,
                        "rmsd": self.rmsd_tensor,
                        "params": n_param_dict,
                    },
                    out_file,
                )
            else:
                # save to a tensor dict
                torch.save(
                    {
                        "x": self.data_tensor,
                        "params": n_param_dict,
                    },
                    out_file,
                )
        return self.data_tensor

    @property
    def read_header(self):
        """Grab metadata which will be used to extract information

        RETURNS
        -------
            total_bytes_to_skip: int
                The total number of bytes that we can read from the file
                to skip the header informatino

            avail_vals: int
                Then number of values that needs to be captured per sample
        """
        # init the byte holding amount
        total_bytes_to_skip = 0

        # init the amount of values in total that should are available
        avail_vals = 0

        # fill in metadata_info
        with open(self.binary_file, 'rb') as bf:

            # loop through known header format information
            for i, entry in enumerate(self.format):

                # Pull out specific amount of bytes specified in format entry tuple
                chunk = bf.read(entry[1])

                # Assign the correct decoded value from the header for each of the param options
                self.metadata_info[self.param_set[i]] = struct.unpack(entry[0], chunk)[0]

                # total bytes to skip
                total_bytes_to_skip += entry[1]
                if self.param_set[i].endswith("_n"):
                    avail_vals += self.metadata_info[self.param_set[i]]

                # Also, add the extra information from last info
                if self.param_set[i] == "bytes_to_skip":
                    total_bytes_to_skip += self.metadata_info[self.param_set[i]]

        return total_bytes_to_skip, avail_vals

    @property
    def binary_data_format(self):
        """Generate format based upon input header data"""
        # init holding information
        data_format = list()

        # init number of input values
        num_values = 0

        # go through available data and create information
        for param in self.param_set:

            # only grab number information
            if param.endswith("_n"):
                data_format.extend([("d", 8)] * self.metadata_info[param])
                num_values += self.metadata_info[param]
        return data_format, num_values

    @property
    def rmsd_calc(self):
        """Fill in RMSD Matrix from passed rmsd_binary"""
        # open file
        with open(self.rmsd_binary, 'rb') as in_rmsd_file:

            # Grab headers
            header_major = in_rmsd_file.read(8)
            header_minor = in_rmsd_file.read(8)

            # read in info
            major_int = str(struct.unpack("Q", header_major)[0])
            minor_int = str(struct.unpack("Q", header_minor)[0])

            assert major_int + "." + minor_int == "1.1", "Version Number Doesn't Match!! 1.1 was asked for!"

            # grab row and column numbers
            row_n_binary = in_rmsd_file.read(8)
            col_n_binary = in_rmsd_file.read(8)

            # conver to int
            row_n = struct.unpack("Q", row_n_binary)[0]
            col_n = struct.unpack("Q", col_n_binary)[0]

            
            # Set up the rmsd pre matrix
            rmsd_tensor = torch.zeros((row_n, col_n), dtype=torch.float32)

            # row track & col track
            row_idx = 0
            col_idx = 1

            # running_max
            running_max = col_n

            # set file read
            file_read = True

            # init time
            dt0 = time.time()

            # Keep looking
            while file_read:

                # float32 information binary
                binary_entry = in_rmsd_file.read(4*(running_max-1))

                # check if the file ended
                if (not binary_entry):
                    # break while loop
                    file_read = False

                # convert to float from binary
                rmsd_value = torch.tensor(struct.unpack("f"*(running_max-1), binary_entry))

                # now assign to 2 entries
                rmsd_tensor[row_idx, col_idx:] = rmsd_value
                rmsd_tensor[col_idx:, row_idx] = rmsd_value

                # col_idx update
                col_idx += 1
                row_idx += 1
                running_max -= 1
                # if col_idx >= col_n:
                #
                #     # Update all tracking matrix info
                #     row_idx += 1
                #     col_idx = row_idx + 1
                #
                if row_idx % 10000 == 0: 
                    print(f"Row: {row_idx}, Time: {round(time.time() - dt0,2)}")

        return rmsd_tensor







##################################################################################################################
#                                               Main
##################################################################################################################
if __name__ == "__main__":
    # If main then test out class

    # explicit file
    file = "/mnt/home/vmulligan/ceph/2024_02_02_polygly_macrocycle_conf_embedding/8mer/organic/8mer_organic_dftb_smd_cartmin_geoopt/training_data/dof_vectors.dat"
    rmsd_file = "/mnt/home/vmulligan/ceph/2024_02_02_polygly_macrocycle_conf_embedding/8mer/organic/8mer_organic_dftb_smd_cartmin_geoopt/training_data/rmsd_matrix.dat"

    # init class
    dofclass = DoFBinaryReader(file, rmsd_file, True, True)

    # check out tensor
    out = dofclass.extract_tensor
    print(out)
