import hashlib
import subprocess

import yaml

from prodml import train
from prodml.config import settings


def test_git_commit_in_this_repo() -> None:
    value = train.git_commit()
    assert value == "unknown" or len(value) == 40


def test_git_commit_falls_back_when_git_fails(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(train.subprocess, "check_output", boom)
    assert train.git_commit() == "unknown"


def test_data_version_reads_the_dvc_lock(tmp_path, monkeypatch) -> None:
    lock = {
        "stages": {
            "build_manifest": {"outs": [{"path": "data/manifest.json", "md5": "abcdef1234567890"}]}
        }
    }
    (tmp_path / "dvc.lock").write_text(yaml.safe_dump(lock))
    monkeypatch.chdir(tmp_path)
    assert train.data_version() == "abcdef123456"


def test_data_version_falls_back_to_a_file_hash(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "manifest_path", manifest)
    assert train.data_version() == hashlib.md5(b"[]").hexdigest()[:12]
