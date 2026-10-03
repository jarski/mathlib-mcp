"""Consistency of the packaged data with the curation files and the sources."""

import json
import tomllib
from pathlib import Path

import pytest

from mathlib_mcp.cards import load_cards

REPO = Path(__file__).resolve().parents[1]


def curated_ids() -> set[str]:
    ids = set()
    for path in (REPO / "curation" / "cards").glob("*.toml"):
        ids |= set(tomllib.loads(path.read_text())["cards"])
    return ids


def test_every_curated_entry_is_in_the_packaged_data():
    cards = load_cards()
    aliases = {a for c in cards.values() for a in c.aliases}
    for card_id in curated_ids():
        assert card_id in cards or card_id in aliases, card_id
        assert (cards.get(card_id) or next(c for c in cards.values()
                                           if card_id in c.aliases)).curated


def test_curated_cards_have_backend_and_pitfalls():
    for card in load_cards().values():
        if card.curated:
            assert card.backend, card.id
            assert card.pitfalls, card.id


@pytest.mark.skipif(not (REPO / ".sources").is_dir(), reason="library sources not fetched")
def test_packaged_data_is_up_to_date(tmp_path):
    from mathlib_mcp.extract import build

    build.main(["--out", str(tmp_path)])
    packaged = REPO / "src" / "mathlib_mcp" / "data"
    for name in ("cards.json", "symbols.json", "llms.txt"):
        fresh, current = (tmp_path / name).read_text(), (packaged / name).read_text()
        if name.endswith(".json"):
            assert json.loads(fresh) == json.loads(current), f"{name} is stale: run the build"
        else:
            assert fresh == current, f"{name} is stale: run the build"
