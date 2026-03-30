from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_dirs() -> dict[str, Path]:
    root = project_root()
    paths = {
        "ROOT": root,
        "RAW": root / "data" / "raw",
        "PROCESSED": root / "data" / "processed",
        "MODELS": root / "models",
        "CHARTS": root / "charts",
        "REPORTS": root / "reports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
