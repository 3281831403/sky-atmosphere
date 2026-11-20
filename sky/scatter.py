"""Atmospheric scattering primitives for a dynamic sky renderer.

Units and tolerances
--------------------
* Wavelengths are nanometres (nm).
* Altitudes are kilometres (km); the model uses a scale height of 8 km.
* Colours are linear-light RGB components in the range 0.0 .. 1.0.
* Results are deterministic and are checked with a tolerance of 1e-6.
"""

from __future__ import annotations

import math

# Scale height of an exponential atmosphere, in metres.
_SCALE_HEIGHT_M = 8000.0

# Rayleigh reference coefficient at 550 nm, in m^2/kg.
_RAYLEIGH_REF = 1.0e-5

# Mie reference coefficient at 550 nm for unit turbidity, in m^2/kg.
_MIE_REF = 2.0e-6

# Angstrom exponent of the Mie wavelength dependence.
_MIE_ANGSTROM = 1.3

# Henyey-Greenstein anisotropy used by the sky model.  Must stay strictly
# inside (-1, 1): the endpoints make the phase function singular.
_MIE_G = 0.76

# RGB wavelengths in nanometres, ordered red, green, blue.
_CHANNELS_NM = (650.0, 550.0, 450.0)

# Artistic exposure applied to the (dimensionless) single-scattering integral.
_SKY_EXPOSURE = 40.0


class SkyError(Exception):
    """Raised when atmospheric parameters are outside the supported domain."""


def _require_finite(value, name: str) -> float:
    """Return ``value`` as a float, raising SkyError unless it is finite."""
    if isinstance(value, bool):
        raise SkyError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SkyError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise SkyError(f"{name} must be finite")
    return number


def _require_direction(direction, name: str) -> tuple[float, float]:
    """Validate an ``(azimuth_deg, elevation_deg)`` pair."""
    try:
        azimuth, elevation = direction
    except (TypeError, ValueError) as exc:
        raise SkyError(f"{name} must be an (azimuth, elevation) pair") from exc
    return (
        _require_finite(azimuth, f"{name} azimuth"),
        _require_finite(elevation, f"{name} elevation"),
    )


def _direction_vector(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Convert azimuth/elevation in degrees to a unit vector."""
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cos_elevation = math.cos(elevation)
    return (
        cos_elevation * math.cos(azimuth),
        cos_elevation * math.sin(azimuth),
        math.sin(elevation),
    )


def _cosine_between(view_dir, sun_dir) -> float:
    """Return the cosine of the angle between two direction pairs."""
    view = _direction_vector(*view_dir)
    sun = _direction_vector(*sun_dir)
    cosine = sum(a * b for a, b in zip(view, sun))
    # Guard against tiny floating-point overshoots at the domain edges.
    return max(-1.0, min(1.0, cosine))


def rayleigh_coeff(wavelength_nm: float) -> float:
    """Return the Rayleigh scattering coefficient at ``wavelength_nm``.

    Contract
    --------
    * ``wavelength_nm`` is expressed in nanometres (550.0 means 550 nm).
    * The result is a mass extinction coefficient in m^2/kg: positive and
      proportional to ``wavelength_nm ** -4``.
    * The coefficient is normalised to ``1.0e-5`` m^2/kg at 550 nm:
      ``1.0e-5 * (550.0 / wavelength_nm) ** 4``.
    * Red light therefore has the smallest coefficient.
    * Non-finite wavelengths or wavelengths <= 0 raise :class:`SkyError`.
    """
    wavelength = _require_finite(wavelength_nm, "wavelength_nm")
    if wavelength <= 0.0:
        raise SkyError("wavelength_nm must be > 0")
    return _RAYLEIGH_REF * (550.0 / wavelength) ** 4


def mie_coeff(wavelength_nm: float, turbidity: float) -> float:
    """Return the Mie scattering coefficient for the given turbidity.

    Contract
    --------
    * ``wavelength_nm`` is expressed in nanometres (550.0 means 550 nm).
    * ``turbidity`` is dimensionless and must be finite and >= 0.
    * The result is a mass extinction coefficient in m^2/kg:
      ``2.0e-6 * turbidity * (550.0 / wavelength_nm) ** 1.3``.
    * It grows linearly with turbidity, and short wavelengths are scattered
      more strongly than long wavelengths (Ångström exponent 1.3).
    * ``turbidity == 0`` returns 0.0.
    * Invalid wavelengths or turbidity values raise :class:`SkyError`.
    """
    wavelength = _require_finite(wavelength_nm, "wavelength_nm")
    if wavelength <= 0.0:
        raise SkyError("wavelength_nm must be > 0")
    turbidity = _require_finite(turbidity, "turbidity")
    if turbidity < 0.0:
        raise SkyError("turbidity must be >= 0")
    return _MIE_REF * turbidity * (550.0 / wavelength) ** _MIE_ANGSTROM


def phase_rayleigh(cos_theta: float) -> float:
    """Return the Rayleigh phase function at ``cos_theta``.

    Contract
    --------
    * ``cos_theta`` is dimensionless and must lie in [-1.0, 1.0].
    * The result is ``3.0 / (16.0 * pi) * (1.0 + cos_theta ** 2)``.
    * The function is symmetric around zero and reaches its maximum at
      ``abs(cos_theta) == 1.0``.
    * Values outside the domain raise :class:`SkyError`.
    """
    cosine = _require_finite(cos_theta, "cos_theta")
    if cosine < -1.0 or cosine > 1.0:
        raise SkyError("cos_theta must lie in [-1, 1]")
    return 3.0 / (16.0 * math.pi) * (1.0 + cosine * cosine)


def phase_mie(cos_theta: float, g: float) -> float:
    """Return the Henyey-Greenstein Mie phase function.

    Contract
    --------
    * ``cos_theta`` is dimensionless and must lie in [-1.0, 1.0].
    * ``g`` is the dimensionless anisotropy factor and must satisfy
      ``-1.0 < g < 1.0``; endpoints are singular and must raise
      :class:`SkyError`.
    * The result is
      ``(1.0 - g ** 2) / (4.0 * pi * (1.0 + g ** 2 - 2.0 * g * cos_theta) ** 1.5)``.
    * ``g == 0`` degenerates to the isotropic value ``1.0 / (4.0 * pi)``.
    * Values outside either domain raise :class:`SkyError`.
    """
    cosine = _require_finite(cos_theta, "cos_theta")
    if cosine < -1.0 or cosine > 1.0:
        raise SkyError("cos_theta must lie in [-1, 1]")
    anisotropy = _require_finite(g, "g")
    if anisotropy <= -1.0 or anisotropy >= 1.0:
        raise SkyError("g must lie strictly inside (-1, 1)")
    denominator = 1.0 + anisotropy * anisotropy - 2.0 * anisotropy * cosine
    return (1.0 - anisotropy * anisotropy) / (
        4.0 * math.pi * denominator ** 1.5
    )


def optical_depth(view_dir, sun_dir, density: float) -> float:
    """Integrate atmospheric optical depth along a view ray.

    Contract
    --------
    * ``view_dir`` and ``sun_dir`` are ``(azimuth_deg, elevation_deg)`` pairs.
      Elevation is measured in degrees above the horizon.
    * ``density`` is the sea-level density in kg/m^3 and must be finite and
      >= 0.
    * The atmosphere is exponential with scale height ``H = 8.0`` km
      (8000 m) and altitude measured in km:
      ``rho(h) = density * exp(-h / H)``.
    * For a plane-parallel ray at elevation ``e`` the depth is the mass column
      ``density * 8000.0 / sin(radians(e))`` in kg/m^2.  ``e`` is clamped to
      [0.5, 90] degrees for numerical stability, so the horizon direction has
      the largest finite depth and the zenith has the smallest.
    * Multiplying this mass column by a coefficient from :func:`rayleigh_coeff`
      or :func:`mie_coeff` (m^2/kg) gives a dimensionless optical depth.
    * ``sun_dir`` is accepted for call-site symmetry; extinction along the
      view ray depends only on ``view_dir`` and ``density``.
    * Invalid direction or density values raise :class:`SkyError`.
    """
    _, elevation = _require_direction(view_dir, "view_dir")
    _require_direction(sun_dir, "sun_dir")
    density = _require_finite(density, "density")
    if density < 0.0:
        raise SkyError("density must be >= 0")
    clamped_elevation = max(0.5, min(90.0, elevation))
    return density * _SCALE_HEIGHT_M / math.sin(math.radians(clamped_elevation))


def _channel_radiance(
    wavelength_nm: float,
    cosine: float,
    view_mass: float,
    sun_mass: float,
    density: float,
    turbidity: float,
) -> float:
    """Single-scattering radiance for one wavelength channel."""
    beta_rayleigh = rayleigh_coeff(wavelength_nm)
    beta_mie = mie_coeff(wavelength_nm, turbidity)
    beta_total = beta_rayleigh + beta_mie
    if beta_total <= 0.0:
        return 0.0

    phase_r = phase_rayleigh(cosine)
    phase_m = phase_mie(cosine, _MIE_G)

    # Extinction along the view ray and along the sun ray to the scattering
    # point.  Both use the exponential mass column, so a low sun is reddened.
    view_transmittance = math.exp(-beta_total * view_mass)
    sun_transmittance = math.exp(-beta_total * sun_mass)

    # Single scattering integrates beta * exp(-beta s) to (1 - T_view), so the
    # channel weight is beta_channel / beta_total.
    scattered = (1.0 - view_transmittance) * sun_transmittance
    return scattered * (
        (beta_rayleigh / beta_total) * phase_r
        + (beta_mie / beta_total) * phase_m
    )


def scatter_color(view_dir, sun_dir, params: dict) -> list[float]:
    """Return the linear RGB sky colour for one view direction.

    Contract
    --------
    * ``view_dir`` and ``sun_dir`` are ``(azimuth_deg, elevation_deg)`` pairs.
    * ``params`` must provide ``"density"`` (sea-level kg/m^3) and
      ``"turbidity"`` (dimensionless, >= 0).
    * The view elevation is clamped to [0, 90] degrees; values above the
      zenith are treated as exactly 90 degrees, never as a black band.
    * If the sun elevation is <= 0, direct sunlight is absent and the result
      is exactly ``[0.0, 0.0, 0.0]`` (night sky, not blue).
    * The returned value is three linear RGB components in [0.0, 1.0].
      A high clear sun produces a blue-dominant sky; a low sun near the
      horizon produces a red/orange-dominant sky.
    * The model combines the 650/550/450 nm channels, the Rayleigh and Mie
      coefficients, the phase functions above, and atmospheric
      transmittance.  It is deterministic and contains no random sampling.
    * Invalid arguments raise :class:`SkyError`.
    """
    view_azimuth, view_elevation = _require_direction(view_dir, "view_dir")
    sun_azimuth, sun_elevation = _require_direction(sun_dir, "sun_dir")

    if not isinstance(params, dict):
        raise SkyError("params must be a dictionary")
    try:
        density = _require_finite(params["density"], "density")
        turbidity = _require_finite(params["turbidity"], "turbidity")
    except KeyError as exc:
        raise SkyError(f"params is missing {exc.args[0]!r}") from exc
    if density < 0.0:
        raise SkyError("density must be >= 0")
    if turbidity < 0.0:
        raise SkyError("turbidity must be >= 0")

    # No direct sunlight below the horizon: the sky is night, not blue.
    if sun_elevation <= 0.0:
        return [0.0, 0.0, 0.0]

    # Clamp the view elevation into the supported hemisphere so directions
    # beyond the zenith fold onto the zenith instead of going black.
    view_elevation = max(0.0, min(90.0, view_elevation))
    sun_elevation = max(0.0, min(90.0, sun_elevation))

    clamped_view = (view_azimuth, view_elevation)
    clamped_sun = (sun_azimuth, sun_elevation)

    cosine = _cosine_between(clamped_view, clamped_sun)
    view_mass = optical_depth(clamped_view, clamped_sun, density)
    sun_mass = optical_depth(clamped_sun, clamped_view, density)

    color = []
    for wavelength_nm in _CHANNELS_NM:
        radiance = _channel_radiance(
            wavelength_nm,
            cosine,
            view_mass,
            sun_mass,
            density,
            turbidity,
        )
        color.append(max(0.0, min(1.0, radiance * _SKY_EXPOSURE)))
    return color


def tonemap(color, exposure: float) -> list[float]:
    """Apply a smooth exposure curve to a linear RGB colour.

    Contract
    --------
    * ``color`` is a sequence of three linear components, each in [0.0, 1.0].
    * ``exposure`` is a finite, non-negative EV-style value.
    * With ``q = exp(-exposure)``, every component is mapped by
      ``c / (q + (1.0 - q) * c)``.
    * ``exposure == 0`` is therefore exactly the identity transform.
    * For positive exposure the curve is smooth, monotonic in each component,
      lifts tones toward 1.0, and never hard-clips or returns values above 1.0.
    * Negative or non-finite exposure, invalid colours, or negative colour
      components raise :class:`SkyError`.
    """
    exposure = _require_finite(exposure, "exposure")
    if exposure < 0.0:
        raise SkyError("exposure must be >= 0")

    try:
        components = list(color)
    except TypeError as exc:
        raise SkyError("color must be a sequence of three components") from exc
    if len(components) != 3:
        raise SkyError("color must have exactly three components")

    values = []
    for component in components:
        value = _require_finite(component, "color component")
        if value < 0.0 or value > 1.0:
            raise SkyError("color components must lie in [0, 1]")
        values.append(value)

    q = math.exp(-exposure)
    if q == 0.0:  # exp underflowed for a very large exposure
        return [1.0 if value > 0.0 else 0.0 for value in values]

    return [value / (q + (1.0 - q) * value) for value in values]


def sky_batch(samples, params: dict) -> dict:
    """Render a batch of sky samples and return their colours in order.

    Contract
    --------
    * ``samples`` is an iterable of dictionaries.  Each sample must contain a
      ``"view_dir"`` ``(azimuth_deg, elevation_deg)`` pair and may contain an
      optional ``"sun_dir"`` pair that overrides ``params["sun_dir"]``.
    * ``params`` must provide ``"sun_dir"``, ``"density"``, and
      ``"turbidity"``.
    * The return value is ``{"colors": [...]}``: one linear RGB list per
      sample, in the same order as ``samples``.
    * An empty batch returns ``{"colors": []}``.
    * If any sample or parameter is invalid, the whole call raises
      :class:`SkyError` and produces no partial result.
    * Samples are rendered independently; no accumulation buffer or other
      mutable state may be shared between them.
    """
    if not isinstance(params, dict):
        raise SkyError("params must be a dictionary")
    try:
        default_sun = params["sun_dir"]
        density = params["density"]
        turbidity = params["turbidity"]
    except KeyError as exc:
        raise SkyError(f"params is missing {exc.args[0]!r}") from exc

    # Validate the shared parameters once, up front, so an empty batch cannot
    # hide an invalid argument.
    default_sun = _require_direction(default_sun, "sun_dir")
    density = _require_finite(density, "density")
    turbidity = _require_finite(turbidity, "turbidity")
    if density < 0.0:
        raise SkyError("density must be >= 0")
    if turbidity < 0.0:
        raise SkyError("turbidity must be >= 0")
    light_params = {"density": density, "turbidity": turbidity}

    try:
        sample_list = list(samples)
    except TypeError as exc:
        raise SkyError("samples must be iterable") from exc

    colors = []
    for sample in sample_list:
        if not isinstance(sample, dict):
            raise SkyError("each sample must be a dictionary")
        try:
            view_dir = sample["view_dir"]
        except KeyError as exc:
            raise SkyError("sample is missing 'view_dir'") from exc
        sun_dir = sample.get("sun_dir", default_sun)
        # scatter_color validates everything; each call has its own state, so
        # no accumulator is shared between samples.
        colors.append(scatter_color(view_dir, sun_dir, light_params))

    return {"colors": colors}
