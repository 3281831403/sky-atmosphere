"""Atmospheric scattering package for the sky-atmosphere exercise."""

from .scatter import (
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

__all__ = [
    "SkyError",
    "mie_coeff",
    "optical_depth",
    "phase_mie",
    "phase_rayleigh",
    "rayleigh_coeff",
    "scatter_color",
    "sky_batch",
    "tonemap",
]
