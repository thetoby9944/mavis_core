from pathlib import Path
import numpy as np


def maybe_new_dir(path: Path, dir_name=None, new_dir=False) -> Path:
    folder = path if path.is_dir() else path.parent
    if new_dir:
        (folder / dir_name).mkdir(parents=True, exist_ok=True)
        folder = folder / dir_name
    return folder


def identifier(path:str, path_indices, include_stem=True) -> str:
    if not type(path) == str:
        return None

    p = Path(path)
    parts = "_".join(np.array(p.parts)[path_indices])
    parts = (parts and parts + "_") + (p.stem if include_stem else "")
    return parts
