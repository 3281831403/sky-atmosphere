"""Acceptance tests for the sky-atmosphere exercise.

Every test calls at least one function from :mod:`sky.scatter`.  In the
initial state all fifteen tests fail because those functions are stubs.
"""

import math

import pytest

from sky.scatter import (
    SkyError,
    mie_coeff,
    optical_depth,
    phase_mie,
    phase_rayleigh,
    rayleigh_coeff,
    scatter_color,
    sky_batch,
    tonemap,
)


def test_rayleigh_uses_inverse_fourth_power():
    coeff_450 = rayleigh_coeff(450.0)
    coeff_650 = rayleigh_coeff(650.0)

    assert coeff_450 / coeff_650 == pytest.approx(
        (650.0 / 450.0) ** 4, rel=1e-6
    )
    assert rayleigh_coeff(700.0) < rayleigh_coeff(550.0) < coeff_450


def test_wavelength_unit_is_nanometres():
    assert rayleigh_coeff(550.0) == pytest.approx(1.0e-5, rel=1e-6)
    assert mie_coeff(550.0, 1.0) == pytest.approx(2.0e-6, rel=1e-6)
    assert mie_coeff(450.0, 1.0) / mie_coeff(650.0, 1.0) == pytest.approx(
        (650.0 / 450.0) ** 1.3, rel=1e-6
    )
    assert mie_coeff(450.0, 2.0) > mie_coeff(450.0, 1.0)


def test_mie_phase_has_forward_lobe():
    cos_theta = 1.0
    g = 0.8
    expected = (1.0 - g**2) / (
        4.0 * math.pi * (1.0 + g**2 - 2.0 * g * cos_theta) ** 1.5
    )

    assert phase_mie(cos_theta, g) == pytest.approx(expected, rel=1e-6)
    assert phase_mie(cos_theta, g) > 5.0 * phase_rayleigh(cos_theta)
    assert phase_mie(cos_theta, g) > 10.0 * phase_mie(-1.0, g)


def test_mie_rejects_singular_anisotropy():
    for g in (1.0, -1.0, 1.25, -2.0):
        with pytest.raises(SkyError):
            phase_mie(0.5, g)


def test_optical_depth_grows_toward_horizon():
    zenith = optical_depth((0.0, 90.0), (180.0, 45.0), 1.225)
    horizon = optical_depth((0.0, 0.0), (180.0, 45.0), 1.225)
    low = optical_depth((0.0, 5.0), (180.0, 45.0), 1.225)
    mid = optical_depth((0.0, 45.0), (180.0, 45.0), 1.225)

    assert horizon > 20.0 * zenith
    assert low > mid > zenith
    assert optical_depth((0.0, 90.0), (180.0, 45.0), 2.45) == pytest.approx(
        2.0 * zenith, rel=1e-6
    )


def test_sun_below_horizon_has_no_direct_term():
    params = {"density": 1.225, "turbidity": 1.0}

    night = scatter_color((0.0, 30.0), (180.0, -8.0), params)
    sunset_edge = scatter_color((0.0, 30.0), (180.0, 0.0), params)
    noon = scatter_color((0.0, 40.0), (180.0, 75.0), params)
    sunset = scatter_color((180.0, 4.0), (180.0, 2.0), params)

    assert night == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert sunset_edge == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert all(0.0 <= value <= 1.0 for value in noon)
    assert all(0.0 <= value <= 1.0 for value in sunset)
    assert noon[2] > noon[0]
    assert sunset[0] > sunset[2]


def test_tonemap_rolls_off_highlights():
    source = [0.25, 0.6, 0.98]
    mapped = tonemap(source, 1.0)

    assert len(mapped) == 3
    assert all(0.0 <= value <= 1.0 for value in mapped)
    assert mapped[0] > source[0] + 0.2
    assert mapped[1] > source[1] + 0.1
    assert mapped[2] > source[2]
    assert mapped[2] < 1.0
    assert mapped[0] < mapped[1] < mapped[2]


def test_tonemap_rejects_negative_exposure():
    with pytest.raises(SkyError):
        tonemap([0.2, 0.3, 0.4], -0.5)


def test_batch_does_not_share_accumulator():
    params = {"sun_dir": (180.0, 45.0), "density": 1.225, "turbidity": 1.0}
    view_a = (10.0, 20.0)
    view_b = (200.0, 60.0)
    samples = [{"view_dir": view_a}, {"view_dir": view_b}]
    light_params = {"density": 1.225, "turbidity": 1.0}

    colors = sky_batch(samples, params)["colors"]
    assert len(colors) == 2
    assert colors[0] == pytest.approx(
        scatter_color(view_a, (180.0, 45.0), light_params), rel=1e-6, abs=1e-9
    )
    assert colors[1] == pytest.approx(
        scatter_color(view_b, (180.0, 45.0), light_params), rel=1e-6, abs=1e-9
    )

    baseline = [list(rgb) for rgb in colors]
    invalid = [samples[0], {"view_dir": (0.0, float("nan"))}]
    with pytest.raises(SkyError):
        sky_batch(invalid, params)

    after = sky_batch(samples, params)["colors"]
    assert len(after) == len(baseline)
    for got, expected in zip(after, baseline):
        assert got == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_view_elevation_clamped_above_zenith():
    params = {"density": 1.225, "turbidity": 1.0}

    over_zenith = scatter_color((0.0, 120.0), (180.0, 60.0), params)
    at_zenith = scatter_color((0.0, 90.0), (180.0, 60.0), params)

    assert over_zenith == pytest.approx(at_zenith, rel=1e-9, abs=1e-12)


def test_phase_rayleigh_max_at_forward():
    samples = [-1.0, -0.5, 0.0, 0.5, 1.0]
    values = [phase_rayleigh(cos_theta) for cos_theta in samples]

    assert phase_rayleigh(1.0) == pytest.approx(
        3.0 / (8.0 * math.pi), rel=1e-6
    )
    assert phase_rayleigh(1.0) == pytest.approx(max(values), rel=1e-12)


def test_mie_phase_isotropic_when_g_zero():
    isotropic = 1.0 / (4.0 * math.pi)

    for cos_theta in (-1.0, -0.5, 0.0, 0.5, 1.0):
        assert phase_mie(cos_theta, 0.0) == pytest.approx(
            isotropic, rel=1e-6
        )


def test_tonemap_identity_at_zero_exposure():
    color = [0.2, 0.45, 0.9]

    assert tonemap(color, 0.0) == pytest.approx(color, rel=1e-6, abs=1e-9)


def test_batch_empty_samples():
    params = {"sun_dir": (180.0, 45.0), "density": 1.225, "turbidity": 1.0}

    assert sky_batch([], params) == {"colors": []}


def test_batch_preserves_order_on_success():
    params = {"sun_dir": (180.0, 45.0), "density": 1.225, "turbidity": 1.0}
    light_params = {"density": 1.225, "turbidity": 1.0}
    samples = [
        {"view_dir": (0.0, 5.0)},
        {"view_dir": (45.0, 30.0)},
        {"view_dir": (180.0, 80.0)},
    ]

    colors = sky_batch(samples, params)["colors"]
    assert len(colors) == 3
    for got, sample in zip(colors, samples):
        expected = scatter_color(
            sample["view_dir"], (180.0, 45.0), light_params
        )
        assert got == pytest.approx(expected, rel=1e-6, abs=1e-9)
