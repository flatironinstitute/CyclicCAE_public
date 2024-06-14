import torch
import matplotlib.pyplot as plt

def torsion_extract(L: int, root: int):
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

    print("ROOT INDEX", root_index)
    print("pre_root_index", pre_root_index)
    # phi/psi/omega from root on
    phi_post = indices[root_index + 2::gly_default_natoms].tolist()
    psi_post = indices[root_index::gly_default_natoms].tolist()
    omega_post = indices[root_index+1::gly_default_natoms].tolist()
    # phi_post = phi_post[:-1]
    psi_post = psi_post[:-1]
    omega_post = omega_post[:-1]
    print("phi, psi, omega", phi_post, psi_post, omega_post)

    # phi/psi/omega pre root
    phi_pre = [i for i in range(pre_root_index-5, start_residue_natoms-1, -7)] + [2]
    psi_pre = [i for i in range(pre_root_index-7, start_residue_natoms-1, -7)] + [0]
    omega_pre = [i for i in range(pre_root_index-6, start_residue_natoms-1, -7)] + [1]
    print("PRE phi, psi, omega", phi_pre, psi_pre, omega_pre)

    # generate tensors that will be used 
    torsion_pre = sorted(psi_pre + phi_pre) 
    torsion_post = sorted(psi_post + phi_post)
    psi_end = (L - end_residue_natoms) + 3
    phi_1 = start_residue_natoms - 1
    torsion_idx = [phi_1] + torsion_pre + torsion_post + [psi_end]
    omega_idx = torch.tensor(omega_pre[::-1] + omega_post)

    return torch.tensor(torsion_idx), omega_idx


def T(angle: torch.Tensor):
    """Rotation helper function in the x,y plane
    angle: torch.Tensor: bond angles input
    """
    # First expand angle tensor (B) -> (B, 1) 
    expand_angle = angle.unsqueeze(-1)

    #  First row has shape of (B, 3)
    first_row = torch.cat(
        (
            torch.cos(torch.pi-expand_angle),
            -torch.sin(torch.pi-expand_angle),
            torch.zeros(expand_angle.size(0), 1)
        ), dim=-1,
    )
    # Second row has shape of (B, 3)
    second_row = torch.cat(
        (
            torch.sin(torch.pi-expand_angle),
            torch.cos(torch.pi-expand_angle),
            torch.zeros(expand_angle.size(0), 1)
        ), dim=-1,
    )
    # Third row has shape of (B, 3)
    third_row = torch.tensor([0, 0, 1]).unsqueeze(0).repeat(expand_angle.size(0), 1)

    # We combine each (B, 3) to form a final (B, 3, 3)
    combine = torch.cat(
        (
            first_row.unsqueeze(-1),  # (B, 3) -> (B, 3, 1)
            second_row.unsqueeze(-1), # (B, 3) -> (B, 3, 1)
            third_row.unsqueeze(-1),  # (B, 3) -> (B, 3, 1)
        ), dim=-1,
    )

    return combine

def R(angle):
    """Rotation helper function in y,z plane
    angle: torch.Tensor: torsion angles input
    """
    # First expand angle tensor (B) -> (B, 1) 
    expand_angle = angle.unsqueeze(-1)

    #  First row has shape of (B, 3)
    first_row = torch.tensor([1, 0, 0]).unsqueeze(0).repeat(expand_angle.size(0), 1)
    # Second row has shape of (B, 3)
    second_row = torch.cat(
        (
            torch.zeros(expand_angle.size(0), 1),
            torch.cos(torch.pi-expand_angle),
            -torch.sin(torch.pi-expand_angle),
        ), dim=-1,
    )
    # Third row has shape of (B, 3)
    third_row = torch.cat(
        (
            torch.zeros(expand_angle.size(0), 1),
            torch.sin(torch.pi-expand_angle),
            torch.cos(torch.pi-expand_angle),
        ), dim=-1,
    )

    # We combine each (B, 3) to form a final (B, 3, 3)
    combine = torch.cat(
        (
            first_row.unsqueeze(-1),  # (B, 3) -> (B, 3, 1)
            second_row.unsqueeze(-1), # (B, 3) -> (B, 3, 1)
            third_row.unsqueeze(-1),  # (B, 3) -> (B, 3, 1)
        ), dim=-1,
    )
    return combine


def terminal_peptide(
    torsion: torch.Tensor,
    omega: torch.Tensor,
    bond_angles: torch.Tensor,
    bond_lengths: torch.Tensor,
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

    RETURNS
    -------
    """
    # convert angles to radians
    torsion = torch.deg2rad(torsion)
    bond_angles = torch.deg2rad(bond_angles)

    # bond angles
    # NR   = bond_angles[:, 0]
    # CaR  = bond_angles[:, 2]
    # CR   = bond_angles[:, 5]
    # COR  = bond_angles[:, 6]
    # NHR  = bond_angles[:, 1]
    NR_tag   = torch.deg2rad(torch.tensor(121.7)).repeat(bond_angles.size(0))
    CaR_tag  = torch.deg2rad(torch.tensor(111.2)).repeat(bond_angles.size(0))
    CR_tag   = torch.deg2rad(torch.tensor(1116.2)).repeat(bond_angles.size(0))
    COR_tag  = torch.deg2rad(torch.tensor(123.0)).repeat(bond_angles.size(0))
    NHR_tag  = torch.deg2rad(torch.tensor(1119.2)).repeat(bond_angles.size(0))

    # bond lengths
    NCa = bond_lengths[:, 0]
    CaC = bond_lengths[:, 2]
    CN  = bond_lengths[:, 5]
    CO  = bond_lengths[:, 6]
    NH  = bond_lengths[:, 1]

    # CaH = bond_lengths[:, :, 1]

    # Ideal Omega
    omega = torch.deg2rad(omega) #o_t = torch.tensor(torch.pi).repeat(bond_angles.size(0)) #bond_angles[:, 5]
    omega = torch.cat(
        (
            omega,
            torch.tensor(torch.pi).unsqueeze(0).repeat(omega.size(0), 1),
        ), dim=-1,
    )
    # omega = torch.tensor(torch.pi).unsqueeze(0).repeat(bond_angles.size(0), torsion.size(1)//2)
    

    # Number of residues within the peptides
    n = torsion.size(1)//2
    # Init coordinate holding tensor (B, L, 5, 3)
    coordinates = torch.zeros(torsion.size(0), n, 5, 3)
    # N terminal Atom coordinates (B, 3, 1) For a singular input we would use (3, 1)
    N = torch.zeros(torsion.size(0), 3, 1)
    # Rotation matrix (B, 3, 3)
    rotation = torch.eye(3).unsqueeze(0).repeat(torsion.size(0), 1, 1)

    # generate backbone coordinates
    for i in range(n):

        # bond angles
        # NR   = bond_angles[:, 0*i]
        # CaR  = bond_angles[:, 2*i]
        # CR   = bond_angles[:, 5*i]
        # COR  = bond_angles[:, 6*i]
        # NHR  = bond_angles[:, 1*i]
        NR   = NR_tag
        CaR  = CaR_tag
        CR   = CR_tag
        COR  = COR_tag
        NHR  = NHR_tag

        # bond lengths
        NCa = bond_lengths[:, 0*i]
        CaC = bond_lengths[:, 2*i]
        CN  = bond_lengths[:, 5*i]
        CO  = bond_lengths[:, 6*i]
        NH  = bond_lengths[:, 1*i]

        # Rotate and Translate the Hydrogen Atom is in shape (B, 3, 3)
        V_H = torch.cat(
            (
                (NH*torch.cos(torch.pi-NHR)).unsqueeze(0),
                (-NH*torch.sin(torch.pi-NHR)).unsqueeze(0),
                torch.zeros(1, NHR.size(0)),
            ), dim=0,
        ).T.unsqueeze(-1)
        H = N + torch.matmul(rotation, V_H)
        # Update Rotation Matrix (B, 3, 3)
        rotation = rotation @ T(NR) @ R(torsion[:, 2*i])

        # Rotate and Translate the Alpha Carbon Atom
        V_Ca = torch.cat(
            (
                NCa.unsqueeze(0),
                torch.zeros(1, N.size(0)),
                torch.zeros(1, N.size(0)),
             ), dim=0,
        ).T.unsqueeze(-1)
        Ca = N + torch.matmul(rotation, V_Ca)         
        # Update Rotation Matrix
        rotation = rotation @ T(CaR) @ R(torsion[:, (2*i)+1])


        # Rotate and Translate the C' Atom
        V_C = torch.cat(
            (
                CaC.unsqueeze(0),
                torch.zeros(1, Ca.size(0)),
                torch.zeros(1, Ca.size(0)),
            ), dim=0,
        ).T.unsqueeze(-1)
        C = Ca + torch.matmul(rotation, V_C)
        # Update Rotation Matrix
        rotation = rotation @ T(CR) @ R(omega[:, i])

        # Rotate and Translate The Oxyen Atom
        V_O = torch.cat(
            (
                (-CO * torch.cos(torch.pi-COR)).unsqueeze(0),
                (CO*torch.sin(torch.pi-COR)).unsqueeze(0),
                torch.zeros(1, C.size(0)),
            ), dim=0,
        ).T.unsqueeze(-1)
        O = C + torch.matmul(rotation, V_O)

        # Update N terminal atom position
        V_N = torch.cat(
            (
                CN.unsqueeze(0),
                torch.zeros(1, Ca.size(0)),
                torch.zeros(1, Ca.size(0)),
            ), dim=0,
        ).T.unsqueeze(-1)
        N = C + torch.matmul(rotation, V_N)

        # Update Coordinates
        # coordinates[:, 7*i-5, :] = H.transpose(1, 2).squeeze(1)  #-3
        # coordinates[:, 7*i-4, :] = Ca.transpose(1, 2).squeeze(1) #-4
        # coordinates[:, 7*i-1, :] = C.transpose(1, 2).squeeze(1)  #-1
        # coordinates[:, 7*i, :] = O.transpose(1, 2).squeeze(1)    # 0
        coordinates[:, i, 4, :] = H.transpose(1, 2).squeeze(1)  #-3
        coordinates[:, i, 0, :] = Ca.transpose(1, 2).squeeze(1) #-4
        coordinates[:, i, 1, :] = C.transpose(1, 2).squeeze(1)  #-1
        coordinates[:, i, 3, :] = O.transpose(1, 2).squeeze(1)    # 0

        if i < n:
            # coordinates[:, 7*i+1, :] = N.transpose(1, 2).squeeze(1) #1
            coordinates[:, i, 2, :] = N.transpose(1, 2).squeeze(1) #1

    return coordinates

if __name__ == "__main__":
    # generate values
    # torsion = torch.randint(-180, 180, (10, 7*2))
    # bond_angles = torch.randint(-180, 180, (10, 7*6))
    # bond_lengths = torch.randn((10, 7*6)) + 1

    # Glycine each residue has 7 atoms: N, H, CA, 1HA, 2HA, C, O
    data = torch.load("./binary_import/dof_vectors2024-05-14torch_tensor.pt")
    params = data["params"]
    x = data["x"]

    # pull out values
    torsion = x[:, :params["torsion_main_n"]]
    bond_angles = x[:, params["torsion_main_n"]:params["torsion_main_n"]+params["bond_angle_main_n"]]
    bond_lengths = x[:, params["torsion_main_n"]+params["bond_angle_main_n"]:-params["energy_n"]]

    phi_psi, omega_idx = torsion_extract(torsion.size(-1), 4)
    print("TORSION INDEX", phi_psi)
    print("OMEGA INDEX", omega_idx)

    # index tensors
    omega = torsion[:, omega_idx]
    torsion = torsion[:, phi_psi]
    torsion[:, 0] = torsion[:, 0] + 180
    torsion[:, -1] = torsion[:, -1] + 180
    print("TORSION ANGLES", torsion)
    print("OMEGA ANGLES", omega)


    # grab right atoms
    atom_index = [0, 1, 2, 3, 4, 5, 6] + [i for i in range(9, 57)]
    bond_angles = bond_angles[:, atom_index]
    bond_lengths = bond_lengths[:, atom_index]

    # compute coordinates
    print("TORSION", torsion.shape)
    coordinates = terminal_peptide(torsion, omega, bond_angles, bond_lengths)
    # ca_locations = (torch.arange(8) * 7) + 2
    # ca_locations = torch.tensor([(2*i) for i in range(8)])
    # print("CA LOCATIONS", ca_locations)
    # ca_coords = coordinates[:, ca_locations, :]
    ca_coords = coordinates[:, :, 0, :]
    print(ca_coords.shape)

    terminal_length = torch.norm(ca_coords[:, 0, :] - ca_coords[:, -1, :], dim=-1)
    print(terminal_length)

    
    # Set up 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(ca_coords.size(0)):
        ax.scatter(ca_coords[i, :, 0], ca_coords[i, :, 1], ca_coords[i, :, 2], label=f"Struct {i}", s=20)
        ax.plot(ca_coords[i, :, 0], ca_coords[i, :, 1], ca_coords[i, :, 2])
    plt.legend()
    plt.show()
