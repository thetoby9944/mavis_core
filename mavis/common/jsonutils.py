import json
from pathlib import Path

from mavis.pathlibutils import maybe_new_dir


def save_json(to_save, base_path, new_dir, suffix, dir_name=None):
    p = Path(base_path)
    path = str(maybe_new_dir(p, dir_name, new_dir) / (p.stem + suffix))
    with open(path, "w") as f:
        json.dump(to_save, f)
    return path