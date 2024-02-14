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
    def __init__(self, binary_file: str, save_dataset: bool = False):
        """Custom Binary Rosetta Read In Class

        PARAMS
        ------
            binary_file: str
                Path to binary rosetta/C++ out .dat file

            save_dataset: bool
                Defaults to False, but this is used to specify if you would
                like to torch.save the dataset.
        """
        # save passed args
        self.binary_file = binary_file
        self.save_dataset = save_dataset

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
        print("***EXTRATING HEADER METADATA***")
        self.skip_chunk = self.read_header

        # build extract information data format
        self.data_format, self.num_values = self.binary_data_format

        # create zeros tensor torch 
        self.sample_n = 49700 # this will need to be changed once the information is provided in the header
        self.data_tensor = torch.zeros(size=(self.sample_n, self.num_values))

        # build format given header info for grabbing data
        print("***EXTRACTING BINARY RAW DoF VALUES***")
        with open(self.binary_file, 'rb') as f:
            
            # Skip the amount we need to skip
            f.read(self.skip_chunk)

            # track sample idx
            tensor_sample = 0

            # true for not end
            not_end = True

            # start time 
            dt0 = time.time()

            while not_end:

                # loop through data
                for i, data in enumerate(self.data_format):

                    # binary data extract size
                    binary_data = f.read(data[1])

                    # check if it is empty or not
                    if (not binary_data): #or (tensor_sample == 10000):
                        print(f"PROCESSING ENDS ON SAMPLE {tensor_sample+1}!!!!!!")
                        not_end = False
                        break

                    # unpack information
                    self.data_tensor[tensor_sample, i] = struct.unpack(data[0], binary_data)[0]
                tensor_sample += 1

                # Visual Update on binary read in process
                if tensor_sample % 10000 == 0:
                    print(f"***Sample {tensor_sample} PROCESSED OUT OF {self.sample_n} SAMPLES. ELAPSED TIME {round(time.time() - dt0, 0)}***")


    #################################################################################################################
    #                                        Class Functions
    #################################################################################################################

    @property
    def extract_tensor(self):
        if self.save_dataset:
            out_file_pre = self.binary_file.split("/")[-1].split(".")[0]
            out_file = out_file_pre + "torch_tensor.pt"
            torch.save(
                {
                    "x": self.data_tensor,
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
        """
        # init the byte holding amount
        total_bytes_to_skip = 0

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

                # Also, add the extra information from last info
                if self.param_set[i] == "bytes_to_skip":
                    total_bytes_to_skip += self.metadata_info[self.param_set[i]]

        return total_bytes_to_skip

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



##################################################################################################################
#                                               Main
##################################################################################################################
if __name__ == "__main__":
    # If main then test out class

    # explicit file
    file = "/mnt/home/vmulligan/ceph/2024_02_02_polygly_macrocycle_conf_embedding/8mer/organic/8mer_organic_dftb_smd_cartmin_geoopt/training_data/dof_vectors.dat"

    # init class
    dofclass = DoFBinaryReader(file, True)

    # check out tensor
    out = dofclass.extract_tensor
    print(out)
