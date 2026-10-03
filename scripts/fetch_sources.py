"""Download the pinned library sources that the API data is extracted from.

    uv run python scripts/fetch_sources.py

Reads ``curation/libraries.toml`` and unpacks each library into ``.sources/``. CuPy is
fetched as an sdist; nvmath-python ships only wheels, so a manylinux wheel is used (its
``.py`` files and ``.pyi`` stubs are all the extractor needs). Nothing is imported.
"""

from __future__ import annotations

import io
import json
import tarfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WHEEL_TAG = "cp312-cp312-manylinux_2_28_x86_64"


def release_files(name: str, version: str) -> list[dict]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json") as response:
        return json.load(response)["urls"]


def fetch(name: str, version: str, target: Path) -> None:
    if target.is_dir():
        print(f"{name} {version}: already present in {target}")
        return
    files = release_files(name, version)
    sdists = [f for f in files if f["packagetype"] == "sdist"]
    wheels = [f for f in files if f["filename"].endswith(f"{WHEEL_TAG}.whl")]
    chosen = (sdists or wheels or [None])[0]
    if chosen is None:
        raise SystemExit(f"No sdist or {WHEEL_TAG} wheel for {name} {version}")
    print(f"{name} {version}: downloading {chosen['filename']}")
    with urllib.request.urlopen(chosen["url"]) as response:
        payload = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    if chosen["packagetype"] == "sdist":
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            root = archive.getnames()[0].split("/")[0]
            archive.extractall(target.parent, filter="data")
        (target.parent / root).rename(target)
    else:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(target)


def main() -> None:
    config = tomllib.loads((REPO / "curation" / "libraries.toml").read_text(encoding="utf-8"))
    for library in config["library"]:
        fetch(library["name"], library["version"], REPO / library["source"])


if __name__ == "__main__":
    main()
