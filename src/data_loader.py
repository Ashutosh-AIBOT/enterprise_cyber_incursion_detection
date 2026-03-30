from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]


def _load_nsl_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=NSL_COLUMNS)


def load_nsl_kdd(raw_dir: Path) -> pd.DataFrame:
    train_path = raw_dir / "KDDTrain+.txt"
    test_path = raw_dir / "KDDTest+.txt"

    if train_path.exists() and test_path.exists():
        train_df = _load_nsl_file(train_path)
        test_df = _load_nsl_file(test_path)
        return pd.concat([train_df, test_df], ignore_index=True)

    # Fallback synthetic dataset for pipeline dry-run.
    X, y = make_classification(
        n_samples=12000,
        n_features=35,
        n_informative=14,
        n_redundant=6,
        n_classes=2,
        weights=[0.96, 0.04],
        flip_y=0.01,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["protocol_type"] = np.random.choice(["tcp", "udp", "icmp"], size=len(df), p=[0.7, 0.22, 0.08])
    df["service"] = np.random.choice(["http", "smtp", "dns", "ftp", "ssh"], size=len(df))
    df["flag"] = np.random.choice(["SF", "REJ", "S0"], size=len(df), p=[0.8, 0.1, 0.1])
    df["label"] = np.where(y == 1, "attack", "normal")
    df["difficulty"] = np.random.randint(1, 20, size=len(df))
    return df


def make_binary_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    label_col = "label" if "label" in df.columns else "target"
    y = (df[label_col].astype(str).str.lower() != "normal").astype(int)
    X = df.drop(columns=[c for c in [label_col, "difficulty"] if c in df.columns])
    return X, y
