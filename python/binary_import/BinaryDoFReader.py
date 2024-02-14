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

# import torch
import torch


##################################################################################################################
#                                                Classes
##################################################################################################################

class DoFBinaryReader:
    def __init__(self, binary_file: str):
        """Custom Binary Rosetta Read In Class

        PARAMS
        ------
            binary_file: str
                Path to binary rosetta/C++ out .dat file
        """
        # save passed args
        self.binary_file = binary_file

        # Set storing header information
        # defaultdict to hold information
        self.metadata_info = defaultdict(int)
        # key values that will be used within the newly init defaultdict above
        self.param_set = ["filetype_major", "filetype_minor", "torsion_main_bool", "torsion_side_bool",
                          "bond_angle_main_bool", "bond_angle_side_bool", "bond_length_main_bool", 
                          "bond_length_side_bool", "jump_bool", "energy_bool", "torsion_main_n", 
                          "torsion_side_n", "bond_angle_main_n", "bond_angle_side_n", "bond_length_main_n",
                          "bond_length_side_n", "jump_n", "energy_n", "bytes_to_skip"] 

        # Format to use for unpacking the binary information
        self.format = [("Q", 8), ("Q", 8), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8)]

        # fill in metadata_info
        with open(self.binary_file, 'rb') as bf:

            # loop through known header format information
            for i, entry in enumerate(self.format):

                # Pull out specific amount of bytes specified in format entry tuple
                chunk = bf.read(entry[1])

                # Assign the correct decoded value from the header for each of the param options
                self.metadata_info[self.param_set[i]] = struct.unpack(entry[0], chunk)[0]

            print("Information to skip: ", bf.read(self.metadata_info["bytes_to_skip"]))

            



        print(self.metadata_info)



file = "/mnt/home/vmulligan/ceph/2024_02_02_polygly_macrocycle_conf_embedding/8mer/organic/8mer_organic_dftb_smd_cartmin_geoopt/training_data/dof_vectors.dat"

# header_info = grab_header_information(file)

# Generate a format for the unpack struct information
# format = [("Q", 8), ("Q", 8), ("?", 1)]
# metadata_info = defaultdict(int)
# format = [("Q", 8), ("Q", 8), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("?", 1), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("Q", 8), ("c"*8, 8)]
#
#
# # This is a test for unpacking the information
# with open(file, 'rb') as newfile:
#     # out = newfile.read(1802)
#     # print(out)
#
#     for entry in format:
#         # specify the chunk to grab from format entry
#         chunk = newfile.read(entry[1])
#         # print out for testing
#         # print(chunk)
#         print(struct.unpack(entry[0], chunk))

DoFBinaryReader(file)
