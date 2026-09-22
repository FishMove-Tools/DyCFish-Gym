"""Paired CFD checkpoints, independent of the ROM training script."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


def package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "gymnasium", "torch", "stable-baselines3", "ansys-fluent-core"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def file_digest(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(model_path, stats_path, contract: dict, seed: int,
                   training_steps: int, run_config=None) -> Path:
    model_path, stats_path = Path(model_path), Path(stats_path)
    metadata = {**contract, "seed": seed, "training_steps": training_steps,
                "versions": package_versions(), "run_config": run_config or {},
                "model_file": model_path.name, "model_sha256": file_digest(model_path),
                "normalization_file": stats_path.name,
                "normalization_sha256": file_digest(stats_path),
                "normalization": {"norm_obs": True, "norm_reward_training": True}}
    target = model_path.with_suffix(".metadata.json")
    target.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def validate_bundle(model_path, stats_path, metadata_path, expected: dict) -> dict:
    """Reject absent/mismatched preprocessing or incompatible policy semantics."""
    for label, value in (("model", model_path), ("normalization statistics", stats_path),
                         ("interface metadata", metadata_path)):
        if not Path(value).is_file():
            raise ValueError(f"Missing {label}: {value}. A matching model/stats/metadata bundle is required.")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Interface metadata must be a JSON object.")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Incompatible checkpoint {key}: saved={metadata.get(key)!r}, expected={value!r}. "
                             "Changing a profile or stage requires a compatible model; automatic ROM-to-CFD transfer is not implemented.")
    for key, path in (("model_sha256", model_path), ("normalization_sha256", stats_path)):
        if metadata.get(key) != file_digest(path):
            raise ValueError(f"Checkpoint bundle integrity mismatch for {path}; do not mix model and normalization files.")
    if metadata.get("normalization", {}).get("norm_obs") is not True:
        raise ValueError("This runner requires the training observation normalization statistics.")
    return metadata


def validate_spaces(model, env) -> None:
    import numpy as np
    for name in ("observation_space", "action_space"):
        saved, current = getattr(model, name), getattr(env, name)
        if saved.shape != current.shape or not np.array_equal(saved.low, current.low) or not np.array_equal(saved.high, current.high):
            raise ValueError(f"Incompatible checkpoint {name}; profile changes and cross-stage loading are not automatic.")


def save_bundle(model, env, model_path, stats_path, contract, seed, run_config=None):
    model_path = Path(model_path).with_suffix(".zip")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    env.save(str(stats_path))
    return write_metadata(model_path, stats_path, contract, seed,
                          int(model.num_timesteps), run_config)
