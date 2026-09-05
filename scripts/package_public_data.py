#!/usr/bin/env python3
"""Create the six archives for the unified public dataset record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-release-dir",
        type=Path,
        default=ROOT.parent / "releases/wind-estimation-data-v1.0.0",
    )
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tar_with_pigz(
    source: Path,
    archive_root: str,
    destination: Path,
    *,
    compression_level: int,
    threads: int,
    overwrite: bool,
    links_dir: Path,
) -> None:
    if destination.exists() and not overwrite:
        print(f"Reuse existing archive: {destination.name}")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    destination.unlink(missing_ok=True)

    link = links_dir / archive_root
    link.unlink(missing_ok=True)
    link.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    print(
        f"Packing {archive_root}: {source} -> {destination.name}",
        flush=True,
    )
    tar = subprocess.Popen(
        [
            "tar",
            "--dereference",
            "-cf",
            "-",
            "-C",
            str(links_dir),
            archive_root,
        ],
        stdout=subprocess.PIPE,
    )
    if tar.stdout is None:
        raise RuntimeError("tar stdout pipe was not created")
    try:
        with partial.open("wb") as output:
            pigz = subprocess.Popen(
                [
                    "pigz",
                    f"-{compression_level}",
                    "-p",
                    str(threads),
                ],
                stdin=tar.stdout,
                stdout=output,
            )
            tar.stdout.close()
            pigz_code = pigz.wait()
        tar_code = tar.wait()
    except BaseException:
        tar.kill()
        partial.unlink(missing_ok=True)
        raise
    if tar_code != 0 or pigz_code != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Archive pipeline failed: tar={tar_code}, pigz={pigz_code}"
        )
    partial.replace(destination)
    print(
        f"Completed {destination.name}: "
        f"{destination.stat().st_size / 1024**3:.2f} GiB",
        flush=True,
    )


def load_sources(release_dir: Path) -> dict[str, Path]:
    path = release_dir / ".staging/PACKAGING_SOURCES.local.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run prepare_public_release.py first"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    sources = {name: Path(value) for name, value in raw.items()}
    for name, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"{name}: source is missing: {source}")
    return sources


def update_archive_manifest(release_dir: Path) -> None:
    manifest = release_dir / "metadata/ARCHIVE_MANIFEST.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    archives = release_dir / "archives"
    for row in rows:
        path = archives / row["archive"]
        if path.is_file():
            row["archive_bytes"] = str(path.stat().st_size)
            row["sha256"] = sha256_file(path)
            row["status"] = "packaged_and_verified"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    checksum_path = release_dir / "CHECKSUMS.sha256"
    public_files = sorted(
        [
            *archives.glob("*"),
            release_dir / "README.md",
            release_dir / "DATA_DICTIONARY.md",
            release_dir / "LICENSE_DATA.txt",
            release_dir / "zenodo_metadata.json",
            *sorted((release_dir / "metadata").glob("*")),
        ]
    )
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in public_files:
            if path.is_file() and path != checksum_path:
                handle.write(
                    f"{sha256_file(path)}  "
                    f"{path.relative_to(release_dir).as_posix()}\n"
                )


def main() -> None:
    args = parse_args()
    release_dir = args.data_release_dir.resolve()
    archives = release_dir / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    sources = load_sources(release_dir)
    links_dir = release_dir / ".staging/archive_links"
    if links_dir.exists():
        shutil.rmtree(links_dir)
    links_dir.mkdir(parents=True)

    specifications = [
        (
            "01_rascal_acquisition_json_v1.0.0.tar.gz",
            "rascal_acquisition_json",
        ),
        (
            "02_rascal_training_csv_v1.0.0.tar.gz",
            "rascal_training_csv",
        ),
        (
            "03_rascal_41d_windows_v1.0.0.tar.gz",
            "rascal_41d_windows",
        ),
        (
            "04_main_and_cross_configuration_results_v1.0.0.tar.gz",
            "experiment_evidence",
        ),
        (
            "05_hitl_campaign_v1.0.0.tar.gz",
            "hitl_campaign",
        ),
    ]
    for archive_name, archive_root in specifications:
        tar_with_pigz(
            sources[archive_name],
            archive_root,
            archives / archive_name,
            compression_level=args.compression_level,
            threads=args.threads,
            overwrite=args.overwrite,
            links_dir=links_dir,
        )

    compact_name = "06_minimal_dataset_v1.0.0.zip"
    compact_destination = archives / compact_name
    if compact_destination.exists() and not args.overwrite:
        print(f"Reuse existing archive: {compact_name}")
    else:
        shutil.copy2(sources[compact_name], compact_destination)
        print(
            f"Copied {compact_name}: "
            f"{compact_destination.stat().st_size / 1024**2:.1f} MiB"
        )

    update_archive_manifest(release_dir)
    shutil.rmtree(links_dir)
    total = sum(path.stat().st_size for path in archives.iterdir())
    print(f"Unified data upload total: {total / 1024**3:.2f} GiB")
    print(f"Release directory: {release_dir}")
    print("No files were uploaded or published.")


if __name__ == "__main__":
    main()
