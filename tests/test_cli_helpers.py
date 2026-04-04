from pathlib import Path

import numpy as np

from cryosparc_2d_class_overlay.cli import (
    OverlaySource,
    harmonic_mean,
    normalize_field_label,
    parse_rgb_color,
    rank_micrograph_items,
    resolve_synthetic_background_color,
)


def test_normalize_field_label():
    assert normalize_field_label("J46") == "J46"
    assert normalize_field_label("J 46 / selected") == "J_46_selected"


def test_harmonic_mean():
    assert harmonic_mean([1.0, 1.0]) == 1.0
    assert harmonic_mean([2.0, 4.0]) == 2.6666666666666665
    assert harmonic_mean([0.0, 1.0]) == 0.0


def test_resolve_synthetic_background_color_auto_uses_white_for_dark_overlay():
    background = resolve_synthetic_background_color(
        "auto",
        [parse_rgb_color("black")],
    )
    assert np.allclose(background, parse_rgb_color("white"))


def test_rank_micrograph_items_balanced_prefers_shared_signal():
    sources = [
        OverlaySource("J46", Path("/tmp/J46"), "selected", "black", parse_rgb_color("black"), {}, {}),
        OverlaySource("J98", Path("/tmp/J98"), "selected", "red", parse_rgb_color("red"), {}, {}),
    ]
    items = [
        (Path("mic_a.mrc"), [list(range(80)), list(range(2))]),
        (Path("mic_b.mrc"), [list(range(12)), list(range(12))]),
        (Path("mic_c.mrc"), [list(range(20)), list(range(9))]),
    ]
    ranked = rank_micrograph_items(items, sources, "balanced", 3)
    assert ranked[0][0].name == "mic_c.mrc"
