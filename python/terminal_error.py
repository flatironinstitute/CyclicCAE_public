import torch
import matplotlib.pyplot as plt


def T(angle):
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
    bond_angles: torch.Tensor,
    bond_lengths: torch.Tensor,
    ):
    """Predict terminal peptide bond length given outputs

    PARAMS
    ------
        torsion: torch.Tensor
            Input of torsion angles (non terminal), within input shape of
            (Batch, residue*7)

        bond_angles: torch.Tensor
            Input of bond_angles (non terminal), with an input shape of
            (Batch, Residue*6)

        bond_lengths: torch.Tensor
            Input of bond_lengths (non terminal), with an input shape of
            (Batch, Residue*6)

    RETURNS
    -------
    """
    # convert angles to radians
    torsion = torch.deg2rad(torsion)
    bond_angles = torch.deg2rad(bond_angles)

    # bond angles
    NR   = bond_angles[:, 0]
    CaR  = bond_angles[:, 1]
    CR   = bond_angles[:, 2]
    COR  = bond_angles[:, 3]
    NHR  = bond_angles[:, 4]
    # NR   = torch.deg2rad(torch.tensor(121.7)).repeat(bond_angles.size(0))
    # CaR  = torch.deg2rad(torch.tensor(111.2)).repeat(bond_angles.size(0))
    # CR   = torch.deg2rad(torch.tensor(1116.2)).repeat(bond_angles.size(0))
    # COR  = torch.deg2rad(torch.tensor(123.0)).repeat(bond_angles.size(0))
    # NHR  = torch.deg2rad(torch.tensor(1119.2)).repeat(bond_angles.size(0))

    # bond lengths
    NCa = bond_lengths[:, 0]
    CaC = bond_lengths[:, 1]
    CN  = bond_lengths[:, 2]
    CO  = bond_lengths[:, 3]
    NH  = bond_lengths[:, 4]

    # CaH = bond_lengths[:, :, 1]

    # Ideal Omega
    omega = torch.tensor(torch.pi).repeat(bond_angles.size(0)) #bond_angles[:, 5]

    # Number of residues within the peptides
    n = torsion.size(1)//2
    # Init coordinate holding tensor (B, L, 3)
    coordinates = torch.zeros(torsion.size(0), 7*n, 3)
    # N terminal Atom coordinates (B, 3, 1) For a singular input we would use (3, 1)
    N = torch.zeros(torsion.size(0), 3, 1)
    # Rotation matrix (B, 3, 3)
    rotation = torch.eye(3).unsqueeze(0).repeat(torsion.size(0), 1, 1)

    # generate backbone coordinates
    for i in range(n):

        # Rotate and Translate  H is in shape (B, 3, 3)
        V_H = torch.cat(
            (
                (NH*torch.cos(torch.pi-NHR)).unsqueeze(0),
                (-NH*torch.sin(torch.pi-NHR)).unsqueeze(0),
                torch.zeros(1, NHR.size(0)),
            ), dim=0,
        ).T.unsqueeze(-1)
        H = N + torch.matmul(rotation, V_H)
        # Update Rotation Matrix (B, 3, 3)
        rotation = rotation @ T(NR) @ R(torsion[:, 2*i-1])

        # Rotate and Translate 
        V_Ca = torch.cat(
            (
                NCa.unsqueeze(0),
                torch.zeros(1, N.size(0)),
                torch.zeros(1, N.size(0)),
             ), dim=0,
        ).T.unsqueeze(-1)
        Ca = N + torch.matmul(rotation, V_Ca)         
        # Update Rotation Matrix
        rotation = rotation @ T(CaR) @ R(torsion[:, 2*i])

        # Rotate and Translate 
        V_C = torch.cat(
            (
                CaC.unsqueeze(0),
                torch.zeros(1, Ca.size(0)),
                torch.zeros(1, Ca.size(0)),
            ), dim=0,
        ).T.unsqueeze(-1)
        C = Ca + torch.matmul(rotation, V_C)
        # Update Rotation Matrix
        rotation = rotation @ T(CR) @ R(omega)

        # Rotate and Translate 
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
        coordinates[:, 7*i-5, :] = H.transpose(1, 2).squeeze(1)  #-3
        coordinates[:, 7*i-4, :] = Ca.transpose(1, 2).squeeze(1) #-4
        coordinates[:, 7*i-1, :] = C.transpose(1, 2).squeeze(1)  #-1
        coordinates[:, 7*i, :] = O.transpose(1, 2).squeeze(1)    #0

        if i < n:
            coordinates[:, 7*i+1, :] = N.transpose(1, 2).squeeze(1) #1

    return coordinates

if __name__ == "__main__":
    # generate values
    torsion = torch.randint(-180, 180, (10, 7*2))
    bond_angles = torch.randint(-180, 180, (10, 7*6))
    bond_lengths = torch.randn((10, 7*6)) + 1

    # compute coordinates
    coordinates = terminal_peptide(torsion, bond_angles, bond_lengths)
    ca_locations = torch.tensor([7*i-4 for i in range(7)])
    ca_coords = coordinates[:, ca_locations, :]
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



