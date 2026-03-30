from pathlib import Path
import sys


def setup_paths():
    root = Path.cwd()
    if root.name == "notebooks":
        root = root.parent

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    return {
        "ROOT": root,
        "RAW": root / "data" / "raw",
        "PROCESSED": root / "data" / "processed",
        "MODELS": root / "models",
        "CHARTS": root / "charts",
    }
