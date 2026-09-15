"""Utilitários de linhagem: versão do dado (DVC) e do código (Git)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


def data_md5(dvc_pointer: Path = Path("data/churn.csv.dvc")) -> str:
    """Lê o md5 do ponteiro DVC — identifica a versão exata do dataset."""
    doc = yaml.safe_load(dvc_pointer.read_text())
    return doc["outs"][0]["md5"]


def git_commit() -> str:
    """Hash curto do commit atual — identifica a versão exata do código."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "no-git"
