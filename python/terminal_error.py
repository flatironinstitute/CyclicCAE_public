import torch
import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class TerminalPeptide:
    def __init__(self):
        pass

    def terminal_peptide_loss(self, dof_inputs: torch.Tensor, real_term: torch.Tensor, params: dict, foldtree_root: int, device: str):
        """Calculate N-C peptide bond Loss; Currently using Ideal bond length & bond angle

        PARAMS
        ------
        dof_inputs: Tensor, (N, M)
            model reconstructed DoFs. This includes decomposed angles. This needs to be correctd
        real_term: Tensor, (N, 1)
            Pre-calculated terminal peptide bond length
        params: dict
            How many terms there are for each individual DoF (torsion, bond angle, bond length)
        foldtree_root: int
            Specify the root of the foldetree that the peptide is using

        RETURNS
        -------
        loss : Tensor, float
            Regression based loss 1/n * Sigma_i((x_hat_i - x_i)**2)
        """
        # Define decomp numbers
        decomp_torsion = params["torsion_main_n"]*2
        decomp_bondang = params["bond_angle_main_n"]*2
        # print(decomp_torsion)
        # print(decomp_bondang)

        # grab out terms of interest
        torsion_full = dof_inputs[:, :decomp_torsion]
        bond_angles = dof_inputs[:, decomp_torsion : decomp_torsion + decomp_bondang]
        bond_length = dof_inputs[:, decomp_torsion + decomp_bondang : -params["energy_n"]]
        # print('bond length inputs: ', decomp_torsion + decomp_bondang, ':', -params["energy_n"])
        # 
        # print("IN LOSS FUNC")
        # print(dof_inputs.shape)
        # print(bond_length.shape)
        # print(bond_angles.shape)

        # torsion angles
        torsion_degree = torch.rad2deg( 
            torch.atan2( 
                torsion_full[:, :torsion_full.size(-1)//2],
                torsion_full[:, torsion_full.size(-1)//2:],
            )
        )
        bond_angle_degree = torch.rad2deg( 
            torch.atan2( 
                bond_angles[:, :bond_angles.size(-1)//2],
                bond_angles[:, bond_angles.size(-1)//2:],
            )
        )

        # Extract torsion (phi/psi) and omega index
        phi_psi_idx, omega_idx = self.torsion_extract(torsion_degree.size(-1), foldtree_root)
        # phi_psi_idx, omega_idx = phi_psi_idx.to(device), omega_idx.to(device)

        # grab actual values
        torsion = torsion_degree[:, phi_psi_idx]
        omega = torsion_degree[:, omega_idx]

        # Extract out coordinates
        coordinates = self.terminal_peptide(torsion, omega, bond_angles, bond_length, foldtree_root, device)

        # Extract out N and C
        c_coords = coordinates[:, :, 1, :]
        n_coords = coordinates[:, :, 2, :]

        # roll n coords 1
        n_roll = torch.roll(n_coords, shifts=-1, dims=1)

        # Calculate the n-c peptide length of shape (B)
        bond_pred_length = torch.norm(n_roll - c_coords, dim=-1)
        terminal_pred_length = bond_pred_length[:, -1]
        # print("PRED LENGTH:", terminal_pred_length)

        # perform regression
        loss = torch.pow(terminal_pred_length.view(-1, 1) - real_term.view(-1, 1).to(device), 2)

        return loss

    def terminal_peptide_lengths(self, dof_inputs: torch.Tensor, params: dict, foldtree_root: int, device: str):
        """Calculate N-C peptide bond Loss; Currently using Ideal bond length & bond angle

        PARAMS
        ------
        dof_inputs: Tensor, (N, M)
            model reconstructed DoFs. This includes decomposed angles. This needs to be correctd
        params: dict
            How many terms there are for each individual DoF (torsion, bond angle, bond length)
        foldtree_root: int
            Specify the root of the foldetree that the peptide is using

        RETURNS
        -------
        terminal_pred_length : Tensor, (B)
            Float terminal bond lengths for each sample (B_i)
        """
        # Define decomp numbers
        decomp_torsion = params["torsion_main_n"]
        decomp_bondang = params["bond_angle_main_n"]

        # grab out terms of interest
        torsion_full = dof_inputs[:, :decomp_torsion]
        bond_angles = dof_inputs[:, decomp_torsion : decomp_torsion + decomp_bondang]
        bond_length = dof_inputs[:, decomp_torsion + decomp_bondang : -params["energy_n"]]

        # torsion angles
        # torsion_degree = torch.rad2deg( 
        #     torch.atan2( 
        #         torsion_full[:, :torsion_full.size(-1)//2],
        #         torsion_full[:, torsion_full.size(-1)//2:],
        #     )
        # )
        # bond_angle_degree = torch.rad2deg( 
        #     torch.atan2( 
        #         bond_angles[:, :bond_angles.size(-1)//2],
        #         bond_angles[:, bond_angles.size(-1)//2:],
        #     )
        # )

        # Extract torsion (phi/psi) and omega index
        phi_psi_idx, omega_idx = self.torsion_extract(torsion_full.size(-1), foldtree_root)
        # phi_psi_idx, omega_idx = phi_psi_idx.to(device), omega_idx.to(device)

        # grab actual values
        torsion = torsion_full[:, phi_psi_idx]
        omega = torsion_full[:, omega_idx]

        # Extract out coordinates
        coordinates = self.terminal_peptide(torsion, omega, bond_angles, bond_length, foldtree_root, device)

        # Extract out N and C
        c_coords = coordinates[:, :, 1, :]
        n_coords = coordinates[:, :, 2, :]

        # roll n coords 1
        n_roll = torch.roll(n_coords, shifts=-1, dims=1)

        # Calculate the n-c peptide length of shape (B)
        bond_pred_length = torch.norm(n_roll - c_coords, dim=-1)
        terminal_pred_length = bond_pred_length[:, -1]

        return terminal_pred_length

    def extract_bond_angles(self, dof_length: int, root: int, L: int):
        """Extract bond angle indices

        PARAMS
        ------
        dof_length: int
        root: int
        L: int

        RETURNS
        -------
        bond_angle_idx: (Tensor) (M)
        """
        # Generate a dof number array
        dofs = torch.arange(dof_length)

        for resi in range(L):

            if resi+1 == 1:
                bond_angle_idx = torch.tensor([dofs[-1]-9+3, 0, 1, 3, 6]).unsqueeze(0)
            elif resi+1 < root:
                # Update residue values
                N = bond_angle_idx[0, -1]-4 if resi+1==2 else bond_angle_idx[-1, -1]-4
                Ca = bond_angle_idx[0, -1]+2 if resi+1==2 else bond_angle_idx[-1, -1]+1
                C = bond_angle_idx[0, -1]+3 if resi+1==2 else bond_angle_idx[-1, -1]+2
                O = bond_angle_idx[0, -1]+5 if resi+1==2 else bond_angle_idx[-1, -1]+4
                H = bond_angle_idx[0, -1]+8 if resi+1==2 else bond_angle_idx[-1, -1]+7
                bond_angle_idx = torch.cat(
                    (
                        bond_angle_idx,
                        torch.tensor([N, Ca, C, O, H]).unsqueeze(0),
                    ), dim=0,
                )
            elif resi+1 == root:
                # Update Residues
                N = bond_angle_idx[-1, -1]-4
                Ca = bond_angle_idx[-1, -1]+2
                C = bond_angle_idx[-1, -1]+7
                O = bond_angle_idx[-1, -1]+3
                H = bond_angle_idx[-1, -1]+6
                bond_angle_idx = torch.cat(
                    (
                        bond_angle_idx,
                        torch.tensor([N, Ca, C, O, H]).unsqueeze(0),
                    ), dim=0,
                )
            elif resi+1 == L:
                # Update Residues
                N = bond_angle_idx[-1, -1]+2
                Ca = bond_angle_idx[-1, -1]+3
                C = 0 # Based upon first residue N
                O = bond_angle_idx[-1, -1]+4
                H = bond_angle_idx[-1, -1]+7
                bond_angle_idx = torch.cat(
                    (
                        bond_angle_idx,
                        torch.tensor([N, Ca, C, O, H]).unsqueeze(0),
                    ), dim=0,
                )
            else:
                # Update Residues
                N = bond_angle_idx[-1, -1]+2
                Ca = bond_angle_idx[-1, -1]+3
                C = bond_angle_idx[-1, -1]+8
                O = bond_angle_idx[-1, -1]+4
                H = bond_angle_idx[-1, -1]+7
                bond_angle_idx = torch.cat(
                    (
                        bond_angle_idx,
                        torch.tensor([N, Ca, C, O, H]).unsqueeze(0),
                    ), dim=0,
                )

        return bond_angle_idx.view(-1)


    def torsion_extract(self, L: int, root: int):
        """Extract torsion angles given a peptide size and fold tree

        PARAMS
        ------
        L : int
            Size of the torsion tensor (B, L) (last dimension)
        root: int
            Fold Tree Root for the peptide

        RETURNS
        -------
        torsion_idx: Tensor (N)
            Phi/Psi Torsion locations within torsion tensor
        omega_idx: Tensor (M)
            Omege locations within torsion tensor
        """
        # Make full index tensor
        indices = torch.arange(L)

        # Specify Presets
        start_residue_natoms = 8
        end_residue_natoms = 9
        root_residue_natoms = 6
        gly_default_natoms = 7

        # grab root index
        root_index = start_residue_natoms + gly_default_natoms*(root-2) + root_residue_natoms
        pre_root_index = start_residue_natoms + gly_default_natoms*(root-2)
        # To get the phi residue is
        # phi: pre_root_index - 5
        # omega: pre_root_index - 6
        # psi: pre_root_index - 7

        # print("ROOT INDEX", root_index)
        # print("pre_root_index", pre_root_index)
        # phi/psi/omega from root on
        phi_post = indices[root_index + 2::gly_default_natoms].tolist()
        psi_post = indices[root_index::gly_default_natoms].tolist()
        omega_post = indices[root_index+1::gly_default_natoms].tolist()
        # phi_post = phi_post[:-1]
        psi_post = psi_post[:-1]
        omega_post = omega_post[:-1]
        # print("phi, psi, omega", phi_post, psi_post, omega_post)

        # phi/psi/omega pre root
        phi_pre = [i for i in range(pre_root_index-5, start_residue_natoms-1, -7)] + [2]
        psi_pre = [i for i in range(pre_root_index-7, start_residue_natoms-1, -7)] + [0]
        omega_pre = [i for i in range(pre_root_index-6, start_residue_natoms-1, -7)] + [1]
        phi_pre[0] += 6
        # print("PRE phi, psi, omega", phi_pre, psi_pre, omega_pre)

        # generate tensors that will be used 
        torsion_pre = sorted(psi_pre + phi_pre) 
        torsion_post = sorted(psi_post + phi_post)
        psi_end = (L - end_residue_natoms) + 4
        # phi_1 = start_residue_natoms - 1
        phi_1 = 4
        torsion_idx = [phi_1] + torsion_pre + torsion_post + [psi_end]
        omega_idx = torch.tensor(omega_pre[::-1] + omega_post)

        return torch.tensor(torsion_idx), omega_idx


    def T(self, angle: torch.Tensor, device="cpu"):
        """Rotation helper function in the x,y plane
        angle: torch.Tensor: bond angles input
        """
        # First expand angle tensor (B) -> (B, 1) 
        expand_angle = angle.unsqueeze(-1)

        #  First row has shape of (B, 3)
        first_row = torch.cat(
            (
                torch.cos(torch.pi-expand_angle),
                -1*torch.sin(torch.pi-expand_angle),
                torch.zeros(expand_angle.size(0), 1, dtype=torch.float64, device=device)
            ), dim=-1,
        )
        # Second row has shape of (B, 3)
        second_row = torch.cat(
            (
                torch.sin(torch.pi-expand_angle),
                torch.cos(torch.pi-expand_angle),
                torch.zeros(expand_angle.size(0), 1, dtype=torch.float64, device=device)
            ), dim=-1,
        )
        # Third row has shape of (B, 3)
        third_row = torch.tensor([0, 0, 1], dtype=torch.float64, device=device).unsqueeze(0).repeat(expand_angle.size(0), 1)

        # We combine each (B, 3) to form a final (B, 3, 3)
        combine = torch.cat(
            (
                first_row.unsqueeze(-2),  # (B, 3) -> (B, 1, 3)
                second_row.unsqueeze(-2), # (B, 3) -> (B, 1, 3)
                third_row.unsqueeze(-2),  # (B, 3) -> (B, 1, 3)
            ), dim=-2,
        )

        return combine

    def R(self, angle, device="cpu"):
        """Rotation helper function in y,z plane
        angle: torch.Tensor: torsion angles input
        """
        # First expand angle tensor (B) -> (B, 1) 
        expand_angle = angle.unsqueeze(-1)
        # print('R EXPAND ANGLE', expand_angle[0])

        #  First row has shape of (B, 3)
        # (3) -> (1, 3) -> (B, 3)
        first_row = torch.tensor([1, 0, 0], dtype=torch.float64, device=device).unsqueeze(0).repeat(expand_angle.size(0), 1)
        # Second row has shape of (B, 3)
        second_row = torch.cat(
            (
                torch.zeros(expand_angle.size(0), 1, dtype=torch.float64, device=device), # (B, 1)
                torch.cos(expand_angle),     # (B, 1)
                -1*torch.sin(expand_angle),    # (B, 1)
            ), dim=-1,
        ) # -> (B, 3)
        # Third row has shape of (B, 3)
        third_row = torch.cat(
            (
                torch.zeros(expand_angle.size(0), 1, dtype=torch.float64, device=device), # (B, 1)
                torch.sin(expand_angle),     # (B, 1)
                torch.cos(expand_angle),     # (B, 1)
            ), dim=-1,
        )

        # We combine each (B, 3) to form a final (B, 3, 3)
        combine = torch.cat(
            (
                first_row.unsqueeze(-2),  # (B, 3) -> (B, 3, 1)
                second_row.unsqueeze(-2), # (B, 3) -> (B, 3, 1)
                third_row.unsqueeze(-2),  # (B, 3) -> (B, 3, 1)
            ), dim=-2,
        )

        return combine

    def calculate_angle(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor):
        """Calculate an angle given three points

        PARAMS
        ------
        p1: Tenosor (B, 3)
        p2: Tenosor (B, 3)
        p3: Tenosor (B, 3)
        """
        # Calculate differences
        p12 = p1 - p2
        p23 = p3 - p2

        # Take dot product
        dot_123 = torch.bmm(p12, p23)

        # normalize values
        mag_p12 = torch.norm(p12, dim=-1)
        mag_p23 = torch.norm(p23, dim=-1)

        # generate cos of dot and magnitude
        cos_123 = dot_123 / (mag_p12 * mag_p23)

        # take the acosine of the value to get radians
        rad_123 = torch.acos(cos_123)

        return rad_123


    def terminal_peptide(
        self,
        torsion: torch.Tensor,
        omega: torch.Tensor,
        bond_angles: torch.Tensor,
        bond_lengths: torch.Tensor,
        root_idx: int,
        device: str = "cpu",
        ):
        """Predict terminal peptide bond length given outputs
        Glycine each residue has 7 atoms: N, H, CA, 1HA, 2HA, C, O

        PARAMS
        ------
        torsion: torch.Tensor, (Batch, residue*7)
            Input of torsion angles (non terminal), within input shape of

        bond_angles: torch.Tensor, (Batch, Residue*6)
            Input of bond_angles (non terminal), with an input shape of

        bond_lengths: torch.Tensor, (Batch, Residue*6)
            Input of bond_lengths (non terminal), with an input shape of

        root_idx: int
            The position where the root residue is in Rosetta

        device: str
            cpu or cuda

        RETURNS
        -------
        """
        # convert angles to radians
        torsion = torch.deg2rad(torsion)
        bond_angles = torch.deg2rad(bond_angles)

        # Real Omega
        omega = torch.deg2rad(omega) 
        omega = torch.cat(
            (
                # torch.deg2rad(torch.tensor(-91.3)).unsqueeze(0).repeat(omega.size(0), 1),
                omega,
                torch.tensor(torch.pi, dtype=torch.float64, device=device).unsqueeze(0).repeat(omega.size(0), 1), # Add Extra Pi omega for loop
            ), dim=-1,
        )
        # Ideal Omega
        # omega = torch.tensor(torch.pi).unsqueeze(0).repeat(bond_angles.size(0), torsion.size(1)//2)
        

        # Number of residues within the peptides
        n = torsion.size(1)//2
        # Init coordinate holding tensor (B, L, 5, 3)
        coordinates = torch.zeros(torsion.size(0), n, 5, 3, dtype=torch.float64, device=device)
        # N terminal Atom coordinates (B, 3, 1) For a singular input we would use (3, 1)
        N = torch.zeros(torsion.size(0), 3, 1, dtype=torch.float64, device=device)
        # Rotation matrix (B, 3, 3)
        rotation = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0).repeat(torsion.size(0), 1, 1)
        # bond length and angle position
        index_pos = torch.arange(5)

        # generate backbone coordinates
        for i in range(n):
            # print("RESIDUE:", i)
            # print(bond_angles.shape)
            # print(bond_lengths.shape)


            # update index pos
            # if i == root_idx - 1:
            # if i == 0:
            #     # bond angles
            #     NR   = torch.deg2rad(torch.tensor(121.7, dtype=torch.float64, device=device)).repeat(bond_angles.size(0))#bond_angles[:, index_pos[0]]
            #     CaR  = bond_angles[:, index_pos[1]]
            #     CR   = bond_angles[:, index_pos[2]]
            #     COR  = bond_angles[:, index_pos[3]] #torch.deg2rad(torch.tensor(123.0, dtype=torch.float64, device=device)).repeat(bond_angles.size(0))
            #     NHR  = bond_angles[:, index_pos[4]]
            #     print(torch.deg2rad(torch.tensor(121.7, dtype=torch.float64, device=device)))
            #     print(torch.deg2rad(torch.tensor(111.2, dtype=torch.float64, device=device)))
            #     print(torch.deg2rad(torch.tensor(116.2, dtype=torch.float64, device=device)))
            #     print(torch.deg2rad(torch.tensor(123.0, dtype=torch.float64, device=device)))
            #     print(torch.deg2rad(torch.tensor(119.2, dtype=torch.float64, device=device)))
            #
            #     # Bond Lengths
            #     NCa = torch.tensor(1.458, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 0*i]
            #     CaC = torch.tensor(1.524, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 2*i]
            #     CN  = torch.tensor(1.329, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 5*i]
            #     CO  = torch.tensor(1.231, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 6*i]
            #     NH  = torch.tensor(1.010, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 1*i]
            #     # NCa = bond_lengths[:, index_pos[0]]
            #     # CaC = bond_lengths[:, index_pos[1]]
            #     # CN  = bond_lengths[:, index_pos[2]]
            #     # CO  = bond_lengths[:, index_pos[3]] #torch.tensor(1.231, dtype=torch.float64, device=device).repeat(bond_lengths.size(0))
            #     # NH  = bond_lengths[:, index_pos[4]]
            #     # print(NCa)
            #     # print(CaC)
            #     # print(CN)
            #     # print(CO) 
            #     # print(NH) 
            #     print("i == 0", i, index_pos)
            #     index_pos = torch.arange(index_pos[-1]+1, index_pos[-1]+6)
            # # elif i == root_idx - 2:
            # elif i == n-1:
            #     print("i == n-1", i, index_pos)
            #     # bond angles
            #     NR   = bond_angles[:, index_pos[0]]
            #     CaR  = bond_angles[:, index_pos[1]]
            #     CR   = bond_angles[:, index_pos[2]]
            #     COR  = bond_angles[:, index_pos[3]]
            #     NHR  = bond_angles[:, index_pos[4]]
            #     # Bond Lengths
            #     NCa = torch.tensor(1.458, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 0*i]
            #     CaC = torch.tensor(1.524, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 2*i]
            #     CN  = torch.tensor(1.329, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 5*i]
            #     CO  = torch.tensor(1.231, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 6*i]
            #     NH  = torch.tensor(1.010, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 1*i]
            #     # NCa = bond_lengths[:, index_pos[0]]
            #     # CaC = bond_lengths[:, index_pos[1]]
            #     # CN  = torch.deg2rad(torch.tensor(116.2, dtype=torch.float64, device=device)).repeat(bond_angles.size(0))#bond_lengths[:, index_pos[2]]
            #     # CO  = bond_lengths[:, index_pos[3]]
            #     # NH  = bond_lengths[:, index_pos[4]]
            #     index_pos = torch.arange(index_pos[-1]+1, index_pos[-1]+6)
            # else:
            #     print("Filler", i, index_pos)
            #     # bond angles
            #     NR   = bond_angles[:, index_pos[0]]
            #     CaR  = bond_angles[:, index_pos[1]]
            #     CR   = bond_angles[:, index_pos[2]]
            #     COR  = bond_angles[:, index_pos[3]]
            #     NHR  = bond_angles[:, index_pos[4]]
            #     # Bond Lengths
            #     NCa = torch.tensor(1.458, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 0*i]
            #     CaC = torch.tensor(1.524, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 2*i]
            #     CN  = torch.tensor(1.329, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 5*i]
            #     CO  = torch.tensor(1.231, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 6*i]
            #     NH  = torch.tensor(1.010, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 1*i]
            #     # NCa = bond_lengths[:, index_pos[0]]
            #     # CaC = bond_lengths[:, index_pos[1]]
            #     # CN  = bond_lengths[:, index_pos[2]]
            #     # CO  = bond_lengths[:, index_pos[3]]
            #     # NH  = bond_lengths[:, index_pos[4]]
            #     index_pos = torch.arange(index_pos[-1]+1, index_pos[-1]+6)
            #     print("ELSE", index_pos)

            # Ideal for tetsing
            NR   = torch.deg2rad(torch.tensor(121.7, dtype=torch.float64, device=device)).repeat(bond_angles.size(0))  
            CaR  = torch.deg2rad(torch.tensor(111.2, dtype=torch.float64, device=device)).repeat(bond_angles.size(0))  
            CR   = torch.deg2rad(torch.tensor(116.2, dtype=torch.float64, device=device)).repeat(bond_angles.size(0)) 
            COR  = torch.deg2rad(torch.tensor(123.0, dtype=torch.float64, device=device)).repeat(bond_angles.size(0))  
            NHR  = torch.deg2rad(torch.tensor(119.2, dtype=torch.float64, device=device)).repeat(bond_angles.size(0)) 

            # bond lengths (Ideal for testing)
            NCa = torch.tensor(1.458, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 0*i]
            CaC = torch.tensor(1.524, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 2*i]
            CN  = torch.tensor(1.329, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 5*i]
            CO  = torch.tensor(1.231, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 6*i]
            NH  = torch.tensor(1.010, dtype=torch.float64, device=device).repeat(bond_lengths.size(0)) # bond_lengths[:, 1*i]

            # Rotate and Translate the Hydrogen Atom is in shape (B, 3, 3)
            V_H = torch.cat(
                (
                    (NH*torch.cos(torch.pi-NHR)).unsqueeze(0), # (1, B)
                    (-NH*torch.sin(torch.pi-NHR)).unsqueeze(0), # (1, B)
                    torch.zeros(1, NHR.size(0), dtype=torch.float64, device=device), # (1, B)
                ), dim=0,
            ).T.unsqueeze(-1) # (3, B) -> (B, 3, 1)

            # Rot @ V_H (B, 3, 3) @ (B, 3, 1) -> (B, 3, 1)
            H = N + torch.matmul(rotation, V_H) # (B, 3, 1) + (B, 3, 1) -> (B, 3, 1)
            # Update Rotation Matrix (B, 3, 3)
            rotation = rotation @ self.T(NR, device) @ self.R(torsion[:, 2*i], device) # (B, 3, 3) @ (B, 3, 3) @ (B, 3, 3)
            # print(f'ROT H - {i}', rotation[0])
            

            # Rotate and Translate the Alpha Carbon Atom
            V_Ca = torch.cat(
                (
                    NCa.unsqueeze(0),
                    torch.zeros(1, N.size(0), dtype=torch.float64, device=device),
                    torch.zeros(1, N.size(0), dtype=torch.float64, device=device),
                 ), dim=0,
            ).T.unsqueeze(-1)
            # print(f'RES {i} pre-N', N[0].view(-1))
            Ca = N + rotation @ V_Ca
            # Update Rotation Matrix
            rotation = rotation @ self.T(CaR, device) @ self.R(torsion[:, (2*i)+1],device)
            # print(f'ROT Ca - {i}', rotation[0])


            # Rotate and Translate the C' Atom
            V_C = torch.cat(
                (
                    CaC.unsqueeze(0),
                    torch.zeros(1, Ca.size(0), dtype=torch.float64, device=device),
                    torch.zeros(1, Ca.size(0), dtype=torch.float64, device=device),
                ), dim=0,
            ).T.unsqueeze(-1)
            C = Ca + torch.matmul(rotation, V_C)
            # Update Rotation Matrix
            rotation = rotation @ self.T(CR, device) @ self.R(omega[:, i], device)
            # print(f'ROT C - {i}', rotation[0])

            # Rotate and Translate The Oxyen Atom
            V_O = torch.cat(
                (
                    (-CO * torch.cos(torch.pi-COR)).unsqueeze(0),
                    (CO*torch.sin(torch.pi-COR)).unsqueeze(0),
                    torch.zeros(1, C.size(0), dtype=torch.float64, device=device),
                ), dim=0,
            ).T.unsqueeze(-1)
            O = C + torch.matmul(rotation, V_O)

            # Update N terminal atom position
            V_N = torch.cat(
                (
                    CN.unsqueeze(0),
                    torch.zeros(1, Ca.size(0), dtype=torch.float64, device=device),
                    torch.zeros(1, Ca.size(0), dtype=torch.float64, device=device),
                ), dim=0,
            ).T.unsqueeze(-1)
            N = C + torch.matmul(rotation, V_N)

            # Update Coordinates
            coordinates[:, i, 4, :] = H.transpose(1, 2).squeeze(1)  #-3 (B, 3, 1) -> (B, 1, 3) -> (B, 3)
            coordinates[:, i, 0, :] = Ca.transpose(1, 2).squeeze(1) #-4 (B, 3, 1) -> (B, 1, 3) -> (B, 3)
            coordinates[:, i, 1, :] = C.transpose(1, 2).squeeze(1)  #-1 (B, 3, 1) -> (B, 1, 3) -> (B, 3)
            coordinates[:, i, 3, :] = O.transpose(1, 2).squeeze(1)    # 0 (B, 3, 1) -> (B, 1, 3) -> (B, 3)

            if i < n-1:
                coordinates[:, i+1, 2, :] = N.transpose(1, 2).squeeze(1) #1 (B, 3, 1) -> (B, 1, 3) -> (B, 3)

        return coordinates



if __name__ == "__main__":
    # generate values
    # torsion = torch.randint(-180, 180, (10, 7*2))
    # bond_angles = torch.randint(-180, 180, (10, 7*6))
    # bond_lengths = torch.randn((10, 7*6)) + 1

    # init class
    peptide_term_class = TerminalPeptide()

    # Glycine each residue has 7 atoms: N, H, CA, 1HA, 2HA, C, O
    data = torch.load("./binary_import/dof_vectors2024-05-14torch_tensor.pt")
    params = data["params"]
    x = data["x"]

    # pull out values
    torsion_full = x[:, :params["torsion_main_n"]]
    bond_angles = x[:, params["torsion_main_n"]:params["torsion_main_n"]+params["bond_angle_main_n"]]
    bond_lengths = x[:, params["torsion_main_n"]+params["bond_angle_main_n"]:-params["energy_n"]]

    phi_psi, omega_idx = peptide_term_class.torsion_extract(torsion_full.size(-1), 4)
    # print("TORSION INDEX", phi_psi)
    # print("OMEGA INDEX", omega_idx)

    # index tensors
    omega = torsion_full[:, omega_idx]
    torsion = torsion_full[:, phi_psi]

    pep_len, root_pos = 8, 4

    dof_indices = torch.arange(params["bond_angle_main_n"])

    first_res = torch.tensor([dof_indices[-1]-9+3, 0, 1, 4, 6])
    second_res = torch.tensor(
        [
            first_res[-1]-4,
            first_res[-1]+2,
            first_res[-1]+3,
            first_res[-1]+5,
            first_res[-1]+8,
        ]
    )


    # Compute Bond Length Index 
    ## BEFORE ROOT
    root = pep_len - root_pos # need to define terms
    before_root = torch.tensor([0, 1, 2, 3, 6]).unsqueeze(0).repeat(root-1, 1)
    scale_before = torch.arange(root-1).reshape(root-1, 1) * 7 + 1
    # Scale the first one to not add anything
    scale_before[0] -= 1
    before_root += scale_before

    ## ROOT
    root_tensor = torch.tensor([1,2,3,6]) + before_root[-1, -1]

    ## AFTER ROOT
    after_root = (torch.tensor([1,2,3,4,7]) + root_tensor[-1]).unsqueeze(0).repeat(root, 1)
    scale_after = torch.arange(root).reshape(root, 1) * 7
    after_root += scale_after


    atom_index = torch.cat(
        (
            before_root.view(-1),
            root_tensor,
            after_root.view(-1),
        ), dim=-1,
    )

    bond_angle_idx = peptide_term_class.extract_bond_angles(dof_length=params["bond_angle_main_n"], root=4, L=8)
    print("BOND ANGLE")
    print('-----------')
    print(bond_angle_idx)

    print("ATOM INDEX")
    print('-----------')
    print(atom_index)

    print("TESTING PRE")
    print(torch.deg2rad(bond_angles[:, 51]))
    print(torch.deg2rad(bond_angles[:, 0]))
    print(torch.deg2rad(bond_angles[:, 1]))
    print(torch.deg2rad(bond_angles[:, 3]))
    print(torch.deg2rad(bond_angles[:, 6]))

    bond_angles = bond_angles[:, bond_angle_idx]
    bond_lengths = bond_lengths[:, bond_angle_idx]

    # compute coordinates
    coordinates = peptide_term_class.terminal_peptide(torsion, omega, bond_angles, bond_lengths, root_idx=4)
    # ca_locations = (torch.arange(8) * 7) + 2
    # ca_locations = torch.tensor([(2*i) for i in range(8)])
    # print("CA LOCATIONS", ca_locations)
    # ca_coords = coordinates[:, ca_locations, :]
    ca_coords = coordinates[:, :, 0, :]
    n_coords = coordinates[:, :, 2, :]
    c_coords = coordinates[:, :, 1, :]

    # line draw
    bonds = torch.cat(
        (
        n_coords.unsqueeze(2),
        ca_coords.unsqueeze(2),
        c_coords.unsqueeze(2),
        ), dim=2,
    )
    # print('bonds', bonds.shape)
    B, L, A, _ = bonds.shape
    # bonds = bonds.reshape(B, L, A, 3)
    # print("BONDS", bonds)
    # print(bonds.shape)

    # per residue bonds
    # idx_n = torch.arange(-1, 24, 3)
    # idx_c = torch.arange(0, 24, 3)
    # print('idx_n', idx_n)
    # print('idx_c', idx_c)

    # per_residue = bonds[:, :, 1:, :]
    # B, L, A, C = bonds.shape
    # per_residue = bonds.reshape(B, L*A, C)
    # print('per_residue', per_residue.shape)
    # idx_n = idx_n[1:]
    # n_residue = per_residue[:, idx_n, :]
    # c_residue = per_residue[:, idx_c, :]
    n_roll = torch.roll(n_coords, shifts=-1, dims=1)

    residue_plot = 9

    peptide_bond = []
    for b in range(residue_plot, 10):
        for l in range(L):
            C_coord = c_coords[b, l, :].numpy()
            N_coord = n_roll[b, l, :].numpy()
            peptide_bond.append([C_coord, N_coord])
    line_collection = Line3DCollection(peptide_bond, colors="red", linewidths=2, label="N-C Peptide Bond")

    all_bonds = torch.norm(n_roll - c_coords, dim=-1)
    terminal_length = all_bonds[:, -1]
    print("ALL BOND LENGTHS")
    print(all_bonds.shape)
    print(all_bonds)

    print("Terminal Peptide Length", terminal_length)

    # terminal_loss = peptide_term_class.terminal_peptide_loss(x, terminal_length, params, 4, 'cpu')

    
    # Set up 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(residue_plot,10): #ca_coords.size(0)):
        for j in range(8):
            if j == 0:
                ax.scatter(ca_coords[i, j, 0], ca_coords[i, j, 1], ca_coords[i, j, 2], s=120, c="yellow")

                ax.scatter(n_coords[i, j, 0], n_coords[i, j, 1], n_coords[i, j, 2], s=120,c="yellow")

                ax.scatter(c_coords[i, j, 0], c_coords[i, j, 1], c_coords[i, j, 2], s=120,c="yellow")
            elif j == 7:
                ax.scatter(ca_coords[i, j, 0], ca_coords[i, j, 1], ca_coords[i, j, 2], s=120, c="black")

                ax.scatter(n_coords[i, j, 0], n_coords[i, j, 1], n_coords[i, j, 2], s=120,c="black")

                ax.scatter(c_coords[i, j, 0], c_coords[i, j, 1], c_coords[i, j, 2], s=120,c="black")

            elif j == 1:
                ax.scatter(ca_coords[i, :, 0], ca_coords[i, :, 1], ca_coords[i, :, 2], label="Ca", s=60, c="green")

                ax.scatter(n_coords[i, :, 0], n_coords[i, :, 1], n_coords[i, :, 2], s=60, label="N", c="orange")

                ax.scatter(c_coords[i, :, 0], c_coords[i, :, 1], c_coords[i, :, 2], s=60, label="C", c="purple")
            else:
                ax.scatter(ca_coords[i, :, 0], ca_coords[i, :, 1], ca_coords[i, :, 2], s=60, c="green")

                ax.scatter(n_coords[i, :, 0], n_coords[i, :, 1], n_coords[i, :, 2], s=60, c="orange")

                ax.scatter(c_coords[i, :, 0], c_coords[i, :, 1], c_coords[i, :, 2], s=60, c="purple")
    for i in range(bonds.size(1)):
        if i == 0:
            ax.plot(bonds[residue_plot, i, :, 0], bonds[residue_plot, i, :, 1], bonds[residue_plot, i, :, 2], color="blue", label="Backbone")
        else:
            ax.plot(bonds[residue_plot, i, :, 0], bonds[residue_plot, i, :, 1], bonds[residue_plot, i, :, 2], color="blue")
    ax.add_collection(line_collection)

    plt.legend()
    plt.show()
