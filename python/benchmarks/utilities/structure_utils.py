"""Download deposited structures used by the CyclicCAE benchmarks."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

BENCHMARK_STRUCTURES = (
    ("1SKL", "1skl.pdb"),
    ("6AXI", "6axi.pdb"),
    ("6AZG", "6azg.pdb"),
    ("6BE9", "6be9.pdb"),
    ("6UCX", "6ucx.pdb"),
    ("6WI3", "6wi3.pdb"),
    ("7BTI", "7bti.pdb"),
    ("6XBF", "6xbf_nochanges.pdb"),
)


def _validate_pdb(path: Path) -> None:
    """Raise ``ValueError`` when a download does not contain PDB coordinates."""
    with path.open(errors="replace") as handle:
        if any(line.startswith(("ATOM  ", "HETATM")) for line in handle):
            return
    raise ValueError(f"downloaded file does not contain PDB coordinates: {path}")


def download_pdb_structure(
    pdb_id: str,
    destination: Path,
    *,
    force: bool = False,
) -> bool:
    """Download one RCSB PDB atomically; return whether a file was downloaded."""
    destination = Path(destination)
    if destination.exists() and not force:
        _validate_pdb(destination)
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper()),
        headers={"User-Agent": "CyclicCAE-benchmark"},
    )
    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(response, temporary)
        _validate_pdb(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return True


def download_benchmark_structures(
    output_dir: Path,
    *,
    force: bool = False,
) -> list[tuple[Path, bool]]:
    """Download every deposited structure required to prepare benchmark cases."""
    output_dir = Path(output_dir)
    return [
        (
            output_dir / filename,
            download_pdb_structure(
                pdb_id,
                output_dir / filename,
                force=force,
            ),
        )
        for pdb_id, filename in BENCHMARK_STRUCTURES
    ]
