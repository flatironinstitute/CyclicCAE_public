################################################################################
##
## @file python/binary_import/BinaryDoFReader.py
## @brief Allows import of a set of binary degree-of-freedom vectors and an
## RMSD matrix, and conversion to PyTorch input vectors.
## @author Vikram K. Mulligan (vmulligan@flatironinstitute.org), Center for
## Computational Biology, Flatiron Institute.
##
################################################################################


# import base
import struct
from collections import defaultdict
import time

# import torch
import torch


###### Functions

def grab_header_information(input_file: str):
    """Load in the binary data file, but as a normal file to read the
    first few lines to grab information.
    """
    # open the file and iter
    with open(input_file, 'rb') as file:
        last_line = b''
        header_lines = []
        for line in file:
            line = line.strip()
            print(line)
            time.sleep(1)
            if last_line.endswith(b'reference_pose') and line.startswith(b'\x00'):
                break
            else:
                header_lines.append(line)
            last_line = line

    print(header_lines)
    return header_lines

file = "/mnt/home/vmulligan/ceph/2024_02_02_polygly_macrocycle_conf_embedding/8mer/organic/8mer_organic_dftb_smd_cartmin_geoopt/training_data/dof_vectors.dat"
grab_header_information(file)
