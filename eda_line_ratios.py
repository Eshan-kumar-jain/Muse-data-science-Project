"""
eda_line_ratios.py
===================

Phase-1 EDA / line-ratio layer for the HH399 MUSE cube (Trifid Nebula
irradiated Herbig-Haro jet).

WHAT THIS DOES
--------------
1. Loads the MUSE cube (mpdaf Cube, or a bare NumPy array + FITS header)
   into a NaN-masked float32 flux cube + optional variance cube + a
   wavelength axis rebuilt from WCS keywords.
2. For Halpha, [NII]6583, [SII]6716, [SII]6731 and [OIII]5007: makes a
   continuum-subtracted narrow-band flux map (local continuum estimated
   from off-line windows, small velocity shift applied, peak-snapped to
   correct residual wavelength-calibration offsets).
3. Computes a per-spaxel S/N (from the STAT/variance extension if present,
   else from the continuum RMS) and masks low-S/N spaxels.
4. Builds [NII]/Halpha, ([SII]6716+6731)/Halpha and [SII]6716/6731
   ratio maps, guarding against divide-by-zero / low-S/N denominators.
5. Assembles a tidy per-spaxel pandas DataFrame, prints EDA summaries,
   and saves it to CSV.
6. Saves line maps, ratio maps, ratio histograms and a 2D [NII]/Halpha vs
   [SII]/Halpha diagnostic histogram as PNGs under figures/.
7. Prints a short physical interpretation of the ratio maps.

WHY THINGS ARE DONE THE WAY THEY ARE (for the viva)
----------------------------------------------------
* Air vs vacuum wavelengths: this cube's DATA/STAT headers carry
  CTYPE3='AWAV', i.e. the MUSE pipeline calibrates the wavelength solution
  in AIR wavelength, not vacuum, even though SPECSYS='BARYCENT' fixes the
  reference frame. The line list below is entered as rest VACUUM
  wavelengths (the physically fundamental quantities you'd quote from
  atomic data / NIST) and is converted to air with the standard
  IAU/Morton (2000) dispersion-of-air formula before it is ever compared
  to the cube's wavelength axis. Using the raw "6563, 6583, ..." textbook
  numbers directly against an air-calibrated axis would be *approximately*
  right (those textbook numbers are themselves air values), but doing the
  conversion explicitly from precise vacuum rest wavelengths is the
  defensible, traceable choice and removes an ~1.5-2 A systematic that
  would otherwise bias line-window placement.
* Local continuum subtraction: narrow-band imaging of an IFU cube needs a
  continuum estimate free of neighbouring emission lines. Halpha sits
  between the two [NII] lines (~6548, ~6583 A) and the [SII] doublet
  lines are only ~15 A apart, so symmetric two-sided continuum windows
  are not always available without straddling a neighbouring line -- the
  per-line continuum windows below were chosen by hand to avoid known
  contaminants (see CONFIG['lines']).
* Peak snapping: the pipeline builds a reference spectrum from the
  brightest spaxels (by broadband flux) and finds the local maximum
  within a small tolerance of the predicted (air, velocity-shifted)
  line centre, with a 3-point parabolic sub-pixel refinement. This
  absorbs small residual wavelength-calibration or systemic-velocity
  errors without needing a precise redshift a priori.
* S/N and error propagation: line flux = trapezoidal integral of flux
  over the line window minus (continuum density x window width). Where
  a variance (STAT) extension exists, uncertainties are propagated
  through the same trapezoidal weights; the continuum density itself is
  a median (robust to outliers/cosmic rays), whose variance is
  approximated as (pi/2) x mean-pixel-variance / n_pixels (asymptotic
  relative efficiency of the sample median vs mean for Gaussian noise).
  Without a variance extension, the continuum RMS stands in as the
  per-pixel sigma for both continuum and line-window pixels -- a common
  simplification for narrow-band photometry when no formal noise model
  is available; it is *not* photon-limited on a bright line, so this is
  a documented approximation, not a rigorous measurement.
* Masking: a pixel is treated as bad/no-data if it is non-finite or
  exactly zero (MUSE reduced cubes flag missing data as 0 in DATA and
  0/negative in STAT), matching the module's own convention.

USAGE
-----
    python eda_line_ratios.py

or from a notebook that already has an mpdaf `cube` object:

    import eda_line_ratios as elr
    results = elr.run_pipeline(source=cube)

The module never modifies or overwrites the input cube or FITS file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe; Jupyter's own inline backend still works if already set
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm

try:
    from astropy.io import fits
except ImportError:  # pragma: no cover
    fits = None

try:
    from mpdaf.obj import Cube as MpdafCube
except ImportError:  # pragma: no cover
    MpdafCube = None


# ============================================================================
# CONFIG -- edit this block, nothing else should need to change for routine
# re-runs.
# ============================================================================

CONFIG = dict(
    # Path to the MUSE FITS cube (DATA + STAT extensions expected).
    fits_path="HH399_MUSE_WFM.fits",

    # Where to write eda_spaxels.csv and figures/. None -> next to this
    # script (or the current working directory if run from a notebook).
    output_dir=None,
    csv_name="eda_spaxels.csv",
    figures_dirname="figures",

    # Systemic/jet velocity shift applied to the rest-frame line list
    # before peak-snapping, in km/s (positive = redshift). Leave at 0 and
    # let peak-snapping absorb small offsets unless you have a specific
    # kinematic component you want to target.
    velocity_kms=0.0,

    # How far (Angstrom) either side of the predicted line centre to look
    # for the true peak when snapping. Must stay well inside the gaps to
    # neighbouring lines (NII is ~18 A from Halpha; the SII doublet lines
    # are ~15 A apart), so 4 A leaves comfortable margin.
    snap_tolerance_ang=4.0,

    # Per-spaxel S/N threshold below which a line (and ratios depending on
    # it) is masked to NaN.
    snr_threshold=3.0,

    # Fraction (by broadband/white-light flux) of spaxels used to build the
    # reference spectrum for peak-snapping.
    bright_fraction_for_snap=0.2,

    # Rest-frame VACUUM wavelengths (Angstrom) and the local narrow-band /
    # continuum-window geometry for each line. half_width is the
    # half-width (A) of the line-integration window; cont_windows is a
    # list of (lo, hi) continuum sampling windows (A), defined in the same
    # rest-air frame as the line list and shifted along with the
    # peak-snap correction at run time.
    lines={
        "Halpha": dict(
            rest_vac=6564.61, half_width=4.0,
            cont_windows=[(6520, 6542), (6600, 6615)],
        ),
        "NII_6583": dict(
            rest_vac=6585.27, half_width=3.5,
            # Only a red-side window: the blue side is squeezed between
            # Halpha and this line itself (~18 A total gap) and is not
            # safely clear of both.
            cont_windows=[(6600, 6615)],
        ),
        "SII_6716": dict(
            rest_vac=6718.29, half_width=2.5,
            # Outer (blue) window + a narrow inner window sitting between
            # the two SII lines.
            cont_windows=[(6695, 6708), (6721, 6727)],
        ),
        "SII_6731": dict(
            rest_vac=6732.68, half_width=2.5,
            # Inner window (shared, between the doublet lines) + outer
            # (red) window.
            cont_windows=[(6721, 6727), (6745, 6758)],
        ),
        "OIII_5007": dict(
            rest_vac=5008.24, half_width=4.0,
            # [OIII]4959 sits ~48 A blueward -- the blue window is kept
            # well clear of it.
            cont_windows=[(4980, 4990), (5020, 5030)],
        ),
    },
)

C_KMS = 299792.458  # speed of light, km/s


# ============================================================================
# Loading + masking
# ============================================================================

def wave_from_header(hdr, nwave: int) -> np.ndarray:
    """Rebuild the wavelength axis from FITS WCS keywords (CRVAL3/CD3_3 or
    CDELT3, CRPIX3). Used only when no mpdaf Cube.wave is available."""
    crval3 = hdr.get("CRVAL3")
    crpix3 = hdr.get("CRPIX3", 1.0)
    cd3 = hdr.get("CD3_3", hdr.get("CDELT3"))
    if crval3 is None or cd3 is None:
        raise KeyError(
            "Header is missing CRVAL3 and CD3_3/CDELT3 -- cannot rebuild "
            "the wavelength axis. Pass an explicit `header` with these "
            "keywords, or an mpdaf Cube (which carries its own WCS)."
        )
    idx = np.arange(nwave, dtype=np.float64)
    return crval3 + (idx - (crpix3 - 1.0)) * cd3


def mask_bad(arr: np.ndarray, also_nonpositive: bool = False) -> np.ndarray:
    """Return a float32 copy with NaN in place of non-finite or exactly-zero
    entries (the MUSE pipeline's own no-data flag). `also_nonpositive` is
    used for variance arrays, where <=0 is never physically valid."""
    a = np.asarray(arr, dtype=np.float32).copy()
    bad = ~np.isfinite(a) | (a == 0)
    if also_nonpositive:
        bad |= (a < 0)
    a[bad] = np.nan
    return a


def load_cube(source, header=None, var_array=None):
    """Load flux + variance + wavelength axis from any of:
      - a path to a MUSE FITS cube (str/Path),
      - an already-open mpdaf Cube instance,
      - a bare NumPy ndarray of shape (nwave, ny, nx) -- in which case a
        `header` (astropy Header with CRVAL3/CD3_3 or CDELT3+CRPIX3) must
        be supplied so the wavelength axis can be rebuilt, and
        `var_array` may optionally be supplied.

    Returns (flux, var, wave, src_header):
      flux : float32 ndarray (nwave, ny, nx), NaN for bad/zero pixels
      var  : float32 ndarray or None, same shape, NaN for bad pixels
      wave : float64 ndarray (nwave,), Angstrom, AIR wavelengths (matches
             the cube's own AWAV-calibrated axis)
      src_header : astropy Header or None
    """
    is_mpdaf = (MpdafCube is not None) and isinstance(source, MpdafCube)
    is_path = isinstance(source, (str, Path))
    is_array = isinstance(source, np.ndarray)

    if is_mpdaf:
        c = source
        data = c.data
        flux_raw = data.filled(np.nan) if np.ma.isMaskedArray(data) else np.asarray(data)
        var_raw = None
        if c.var is not None:
            v = c.var
            var_raw = v.filled(np.nan) if np.ma.isMaskedArray(v) else np.asarray(v)
        wave = np.asarray(c.wave.coord(), dtype=np.float64)
        hdr = getattr(c, "data_header", None)

    elif is_path:
        if MpdafCube is not None:
            return load_cube(MpdafCube(str(source)))
        if fits is None:
            raise ImportError(
                "Neither mpdaf nor astropy is available -- cannot read the "
                "FITS cube at all."
            )
        with fits.open(source) as hdul:
            flux_raw = np.asarray(hdul["DATA"].data, dtype=np.float32)
            hdr = hdul["DATA"].header
            var_raw = np.asarray(hdul["STAT"].data, dtype=np.float32) if "STAT" in hdul else None
            wave = wave_from_header(hdr, flux_raw.shape[0])

    elif is_array:
        flux_raw = source
        var_raw = var_array
        if header is None:
            raise ValueError(
                "A bare ndarray was passed as `source`; supply a `header` "
                "(astropy Header with CRVAL3/CD3_3 or CDELT3+CRPIX3, "
                "CUNIT3) so the wavelength axis can be recovered."
            )
        hdr = header
        wave = wave_from_header(hdr, flux_raw.shape[0])

    else:
        raise TypeError(f"Unsupported cube source type: {type(source)!r}")

    flux = mask_bad(flux_raw)
    var = None if var_raw is None else mask_bad(var_raw, also_nonpositive=True)
    return flux, var, wave, hdr


# ============================================================================
# Wavelength helpers: vacuum -> air, Doppler shift, peak snapping
# ============================================================================

def vacuum_to_air(wave_vac_ang):
    """IAU-standard (Morton 2000 / VALD) vacuum-to-air conversion, accurate
    to ~10 m/s across the optical. wave_vac_ang in Angstrom."""
    s = 1.0e4 / np.asarray(wave_vac_ang, dtype=np.float64)
    n = (1.0 + 0.0000834254
         + 0.02406147 / (130.0 - s ** 2)
         + 0.00015998 / (38.9 - s ** 2))
    return wave_vac_ang / n


def doppler_shift(wave_ang, velocity_kms):
    return wave_ang * (1.0 + velocity_kms / C_KMS)


def predicted_center(rest_vac_ang, velocity_kms):
    """Rest-frame vacuum wavelength -> observed-frame air wavelength."""
    return doppler_shift(vacuum_to_air(rest_vac_ang), velocity_kms)


def bright_reference_spectrum(flux, bright_fraction=0.2):
    """Median spectrum of the brightest `bright_fraction` of spaxels (by
    broadband/white-light flux), used as a high-S/N reference for
    peak-snapping. Using the brightest spaxels avoids diagnosing the line
    centre from noise-dominated diffuse regions."""
    white = np.nansum(flux, axis=0)
    finite = np.isfinite(white)
    if not np.any(finite):
        raise ValueError("Cube has no finite spaxels -- cannot build a reference spectrum.")
    thresh = np.nanpercentile(white[finite], 100 * (1 - bright_fraction))
    bright_mask = finite & (white >= thresh)
    return np.nanmedian(flux[:, bright_mask], axis=1)


def parabolic_peak(wave, y, i):
    """3-point parabolic sub-pixel refinement of a discrete peak at index i."""
    if i <= 0 or i >= len(y) - 1:
        return wave[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    if not np.all(np.isfinite([y0, y1, y2])):
        return wave[i]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return wave[i]
    delta = np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0)
    dl = wave[i + 1] - wave[i]
    return wave[i] + delta * dl


def snap_peak(wave, ref_spectrum, predicted_wl, tol_ang):
    """Find the true local emission peak within +/- tol_ang of the
    predicted line centre, with sub-pixel refinement. Falls back to the
    predicted centre if the window is empty or all-NaN (e.g. reference
    spectrum has a gap there)."""
    lo, hi = predicted_wl - tol_ang, predicted_wl + tol_ang
    in_window = (wave >= lo) & (wave <= hi)
    idx = np.where(in_window & np.isfinite(ref_spectrum))[0]
    if idx.size == 0:
        return predicted_wl
    i_peak = idx[np.nanargmax(ref_spectrum[idx])]
    return parabolic_peak(wave, ref_spectrum, i_peak)


# ============================================================================
# Narrow-band line flux + continuum + S/N
# ============================================================================

def trapz_weights(x):
    """Weights w_i such that sum(w_i * f_i) == np.trapz(f, x): dlambda for
    interior points, dlambda/2 at the edges. Needed to propagate variance
    correctly through the same trapezoidal integral used for the flux."""
    w = np.zeros_like(x, dtype=np.float64)
    if len(x) < 2:
        return w
    w[1:-1] = (x[2:] - x[:-2]) / 2.0
    w[0] = (x[1] - x[0]) / 2.0
    w[-1] = (x[-1] - x[-2]) / 2.0
    return w


def integrate_line(flux, var, wave, lo, hi):
    """Trapezoidal integral of flux over [lo, hi] (Angstrom), plus the
    propagated variance if `var` is given. A spaxel is only integrated if
    every pixel in the window is finite -- integration cannot skip
    pixels without biasing the flux, unlike the continuum estimate below
    which can tolerate a few bad pixels via a robust median."""
    idx = np.where((wave >= lo) & (wave <= hi))[0]
    if idx.size < 2:
        raise ValueError(f"Line window [{lo}, {hi}] contains fewer than 2 wavelength samples.")
    w = trapz_weights(wave[idx])
    sub = flux[idx]
    valid = np.all(np.isfinite(sub), axis=0)
    integ = np.nansum(w[:, None, None] * sub, axis=0)
    integ[~valid] = np.nan

    var_integ = None
    if var is not None:
        vsub = var[idx]
        var_integ = np.nansum((w[:, None, None] ** 2) * vsub, axis=0)
        var_integ[~valid] = np.nan

    win_width = wave[idx][-1] - wave[idx][0]
    return integ, var_integ, win_width, w


def robust_continuum(flux, var, wave, windows):
    """Per-spaxel local continuum flux density (median over the given
    windows -- robust to outliers/cosmic rays) plus an estimate of its
    own variance. If a formal variance cube is available, the median's
    variance is approximated as (pi/2) * mean-pixel-variance / n_pixels
    (asymptotic efficiency of the median vs the mean under Gaussian
    noise); otherwise the empirical RMS in the windows plays the role of
    the per-pixel sigma."""
    chunks_f, chunks_v = [], []
    for lo, hi in windows:
        idx = np.where((wave >= lo) & (wave <= hi))[0]
        if idx.size == 0:
            continue
        chunks_f.append(flux[idx])
        if var is not None:
            chunks_v.append(var[idx])
    if not chunks_f:
        raise ValueError(f"No cube wavelength samples fall inside continuum windows {windows}.")

    allf = np.concatenate(chunks_f, axis=0)
    density = np.nanmedian(allf, axis=0)
    rms = np.nanstd(allf, axis=0)
    n_eff = np.sum(np.isfinite(allf), axis=0).astype(np.float64)
    n_eff[n_eff == 0] = np.nan

    if var is not None and chunks_v:
        allv = np.concatenate(chunks_v, axis=0)
        mean_var = np.nanmean(allv, axis=0)
        var_density = (np.pi / 2.0) * mean_var / n_eff
    else:
        var_density = (np.pi / 2.0) * (rms ** 2) / n_eff

    return density, rms, n_eff, var_density


def measure_line(flux, var, wave, ref_spectrum, rest_vac, half_width, cont_windows,
                  velocity_kms, snap_tol):
    """Full per-line pipeline: predicted centre -> peak-snapped observed
    centre -> continuum-subtracted line flux -> propagated noise -> S/N."""
    predicted = predicted_center(rest_vac, velocity_kms)
    observed = snap_peak(wave, ref_spectrum, predicted, snap_tol)
    delta = observed - predicted  # residual calibration/velocity offset

    lo, hi = observed - half_width, observed + half_width
    integ, var_integ, win_width, w_line = integrate_line(flux, var, wave, lo, hi)

    shifted_windows = [(a + delta, b + delta) for a, b in cont_windows]
    cont_density, cont_rms, cont_n, cont_var_density = robust_continuum(
        flux, var, wave, shifted_windows
    )

    cont_contribution = cont_density * win_width
    line_flux = integ - cont_contribution
    var_cont_contribution = (win_width ** 2) * cont_var_density

    if var_integ is not None:
        noise = np.sqrt(var_integ + var_cont_contribution)
    else:
        var_integ_fallback = np.sum(w_line ** 2) * (cont_rms ** 2)
        noise = np.sqrt(var_integ_fallback + var_cont_contribution)

    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.where((noise > 0) & np.isfinite(noise), line_flux / noise, np.nan)

    return dict(
        flux=line_flux, noise=noise, snr=snr,
        predicted_center=predicted, observed_center=observed,
        window=(lo, hi), cont_windows=shifted_windows,
    )


def snr_mask(result, threshold):
    """Return a copy of a measure_line() result with flux/noise/snr set to
    NaN wherever S/N < threshold."""
    bad = ~(result["snr"] >= threshold)
    out = dict(result)
    out["flux"] = np.where(bad, np.nan, result["flux"])
    out["noise"] = np.where(bad, np.nan, result["noise"])
    out["snr"] = np.where(bad, np.nan, result["snr"])
    return out


# ============================================================================
# Ratios
# ============================================================================

def safe_ratio(num_flux, den_flux, den_snr, snr_threshold, extra_mask=None):
    """num_flux / den_flux, guarded against divide-by-zero/negative
    denominators and masked wherever the denominator's own S/N is below
    threshold (a low-S/N denominator makes the ratio meaningless even if
    the numerator itself is fine)."""
    ratio = np.full(np.broadcast(num_flux, den_flux).shape, np.nan, dtype=np.float64)
    good = np.isfinite(num_flux) & np.isfinite(den_flux) & (den_flux > 0) & (den_snr >= snr_threshold)
    if extra_mask is not None:
        good &= extra_mask
    ratio[good] = num_flux[good] / den_flux[good]
    return ratio


def compute_ratios(results, snr_threshold):
    ha, nii, s6716, s6731 = results["Halpha"], results["NII_6583"], results["SII_6716"], results["SII_6731"]

    ratio_nii_ha = safe_ratio(nii["flux"], ha["flux"], ha["snr"], snr_threshold)

    sii_sum_flux = s6716["flux"] + s6731["flux"]  # NaN propagates if either is NaN
    sii_sum_ok = np.isfinite(s6716["flux"]) & np.isfinite(s6731["flux"])
    ratio_sii_ha = safe_ratio(sii_sum_flux, ha["flux"], ha["snr"], snr_threshold, extra_mask=sii_sum_ok)

    # Density-sensitive doublet ratio: both lines individually must clear
    # the S/N threshold, since this ratio is only physically meaningful
    # (and numerically stable) when both members are well detected.
    sii_pair_ok = (s6716["snr"] >= snr_threshold) & (s6731["snr"] >= snr_threshold)
    ratio_sii_doublet = safe_ratio(s6716["flux"], s6731["flux"], s6731["snr"], snr_threshold, extra_mask=sii_pair_ok)

    return dict(
        NII_Ha=ratio_nii_ha,
        SII_Ha=ratio_sii_ha,
        SII6716_6731=ratio_sii_doublet,
    )


# ============================================================================
# DataFrame / EDA
# ============================================================================

def build_dataframe(flux, results, ratios):
    """One row per spaxel that lies within the observed field of view
    (at least one finite spectral sample), regardless of per-line S/N --
    S/N-based masking shows up as NaN in the flux/ratio columns, and is
    summarized separately via valid-vs-masked counts."""
    valid_spaxel = np.any(np.isfinite(flux), axis=0)
    yy, xx = np.where(valid_spaxel)

    data = {"x": xx, "y": yy}
    for name, res in results.items():
        data[f"flux_{name}"] = res["flux"][yy, xx]
        data[f"snr_{name}"] = res["snr"][yy, xx]
    for name, r in ratios.items():
        data[f"ratio_{name}"] = r[yy, xx]

    return pd.DataFrame(data)


def print_eda_summary(df, results, ratios, snr_threshold):
    print("\n" + "=" * 70)
    print("EDA SUMMARY")
    print("=" * 70)

    print(f"\nSpaxels within field of view: {len(df)}")
    for name in results:
        n_good = int((df[f"snr_{name}"] >= snr_threshold).sum())
        print(f"  {name:12s}: {n_good:7d} / {len(df):7d} spaxels above S/N >= {snr_threshold} "
              f"({100 * n_good / max(len(df), 1):5.1f}%)")

    print(f"\nRatio columns valid (finite) counts:")
    for name in ratios:
        n_good = int(df[f"ratio_{name}"].notna().sum())
        print(f"  ratio_{name:16s}: {n_good:7d} / {len(df):7d} "
              f"({100 * n_good / max(len(df), 1):5.1f}%)")

    print("\ndf.describe():")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df.describe())

    ratio_cols = [f"ratio_{name}" for name in ratios]
    print("\nRatio-ratio correlation matrix:")
    with pd.option_context("display.width", 140):
        print(df[ratio_cols].corr())


# ============================================================================
# Figures
# ============================================================================

def _imshow_flux(ax, img, title, cmap="inferno"):
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        ax.set_title(f"{title} (no valid data)")
        ax.axis("off")
        return None
    vmin, vmax = np.nanpercentile(finite, [1, 99.5])
    vmax = max(vmax, vmin + 1e-6)
    disp = np.clip(img, 0, None)  # PowerNorm needs non-negative; display-only clip
    im = ax.imshow(disp, origin="lower", cmap=cmap,
                    norm=PowerNorm(gamma=0.5, vmin=max(vmin, 0), vmax=vmax))
    ax.set_title(title)
    ax.set_xlabel("x [spaxel]")
    ax.set_ylabel("y [spaxel]")
    return im


def _imshow_ratio(ax, img, title, cmap="viridis", vrange=None):
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        ax.set_title(f"{title} (no valid data)")
        ax.axis("off")
        return None
    if vrange is None:
        vmin, vmax = np.nanpercentile(finite, [1, 99])
    else:
        vmin, vmax = vrange
    im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x [spaxel]")
    ax.set_ylabel("y [spaxel]")
    return im


def make_figures(results, ratios, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- line maps ---
    for name, res in results.items():
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = _imshow_flux(ax, res["flux"], f"{name} flux (continuum-subtracted, S/N-masked)")
        if im is not None:
            fig.colorbar(im, ax=ax, label="flux [1e-20 erg/s/cm2]")
        fig.tight_layout()
        fig.savefig(figures_dir / f"map_flux_{name}.png", dpi=150)
        plt.close(fig)

    # --- ratio maps ---
    ratio_specs = {
        "NII_Ha": ("[NII]6583 / Halpha", "magma", None),
        "SII_Ha": ("([SII]6716+6731) / Halpha", "magma", None),
        "SII6716_6731": ("[SII]6716 / 6731 (density-sensitive)", "RdBu_r", (0.3, 1.5)),
    }
    for name, (title, cmap, vrange) in ratio_specs.items():
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = _imshow_ratio(ax, ratios[name], title, cmap=cmap, vrange=vrange)
        if im is not None:
            fig.colorbar(im, ax=ax, label="ratio")
        fig.tight_layout()
        fig.savefig(figures_dir / f"map_ratio_{name}.png", dpi=150)
        plt.close(fig)

    # --- histograms ---
    for name, (title, _, vrange) in ratio_specs.items():
        vals = ratios[name][np.isfinite(ratios[name])]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if vals.size > 0:
            lo, hi = (np.nanpercentile(vals, [0.5, 99.5]) if vrange is None else vrange)
            ax.hist(vals, bins=80, range=(lo, hi), color="steelblue", edgecolor="none")
        ax.set_xlabel(title)
        ax.set_ylabel("spaxel count")
        ax.set_title(f"Histogram: {title}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"hist_ratio_{name}.png", dpi=150)
        plt.close(fig)

    # --- 2D diagnostic: [NII]/Ha vs [SII]/Ha ---
    x = ratios["SII_Ha"]
    y = ratios["NII_Ha"]
    good = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    if good.sum() > 0:
        xlo, xhi = np.nanpercentile(x[good], [0.5, 99.5])
        ylo, yhi = np.nanpercentile(y[good], [0.5, 99.5])
        h = ax.hist2d(x[good], y[good], bins=60, range=[[xlo, xhi], [ylo, yhi]],
                       cmap="viridis", norm=LogNorm())
        fig.colorbar(h[3], ax=ax, label="spaxel count")
    ax.set_xlabel("([SII]6716+6731) / Halpha")
    ax.set_ylabel("[NII]6583 / Halpha")
    ax.set_title("Mini ionization diagnostic: [NII]/Halpha vs [SII]/Halpha")
    fig.tight_layout()
    fig.savefig(figures_dir / "diag_NII_SII_vs_Ha.png", dpi=150)
    plt.close(fig)


# ============================================================================
# Interpretation text
# ============================================================================

def print_interpretation(ratios):
    def pct(a, p):
        v = a[np.isfinite(a)]
        return float(np.nanpercentile(v, p)) if v.size else float("nan")

    nii_med = pct(ratios["NII_Ha"], 50)
    sii_med = pct(ratios["SII_Ha"], 50)
    doublet_med = pct(ratios["SII6716_6731"], 50)
    nii_hi = pct(ratios["NII_Ha"], 90)
    sii_hi = pct(ratios["SII_Ha"], 90)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        f"Across the field, [NII]6583/Halpha has a median of {nii_med:.2f} (90th pct "
        f"{nii_hi:.2f}) and ([SII]6716+6731)/Halpha has a median of {sii_med:.2f} (90th "
        f"pct {sii_hi:.2f}). Low-ionization ratios ([NII]/Ha, [SII]/Ha roughly <0.3-0.5) "
        "typically trace photoionized diffuse/PDR-adjacent gas in the Trifid, whereas "
        "spaxels with elevated [SII]/Ha and [NII]/Ha (approaching or exceeding ~0.5-1) "
        "are the classic shock-excitation signature expected along a Herbig-Haro jet: "
        "shocks boost low-ionization collisionally-excited lines relative to Halpha far "
        "more than photoionization alone does. If the HH399 jet axis stands out in the "
        "[SII]/Ha and [NII]/Ha maps as a spatially coherent ridge of enhanced ratios "
        "against a lower, more uniform diffuse-nebula background, that is direct evidence "
        f"of shock excitation along the jet. The [SII]6716/6731 doublet ratio has a "
        f"median of {doublet_med:.2f}; the doublet spans roughly 0.44 (high-density limit, "
        "electron density above the [SII] critical density, ~a few e3-e4 cm^-3) to ~1.4-1.5 "
        "(low-density limit). A doublet ratio depressed toward the jet relative to the "
        "surrounding diffuse gas would indicate locally enhanced electron density, "
        "consistent with compression in a working-surface/bow-shock jet segment."
    )


# ============================================================================
# Runner
# ============================================================================

def resolve_output_dirs(output_dir):
    if output_dir is not None:
        base = Path(output_dir)
    else:
        try:
            base = Path(__file__).resolve().parent
        except NameError:
            base = Path.cwd()
    base.mkdir(parents=True, exist_ok=True)
    return base


def run_pipeline(source=None, header=None, var_array=None, config=None):
    """Run the full EDA / line-ratio pipeline.

    source defaults to CONFIG['fits_path']. May also be an mpdaf Cube
    instance or a bare ndarray (see load_cube's docstring).
    Returns a dict with flux, var, wave, results (per-line), ratios, df.
    """
    cfg = config or CONFIG
    if source is None:
        source = cfg["fits_path"]

    print(f"Loading cube from: {source!r}")
    flux, var, wave, hdr = load_cube(source, header=header, var_array=var_array)
    print(f"Cube shape: {flux.shape}, wavelength range: "
          f"[{np.nanmin(wave):.2f}, {np.nanmax(wave):.2f}] A "
          f"({'with' if var is not None else 'WITHOUT'} variance extension)")

    ref_spectrum = bright_reference_spectrum(flux, cfg["bright_fraction_for_snap"])

    results = {}
    for name, line_cfg in cfg["lines"].items():
        raw = measure_line(
            flux, var, wave, ref_spectrum,
            rest_vac=line_cfg["rest_vac"],
            half_width=line_cfg["half_width"],
            cont_windows=line_cfg["cont_windows"],
            velocity_kms=cfg["velocity_kms"],
            snap_tol=cfg["snap_tolerance_ang"],
        )
        masked = snr_mask(raw, cfg["snr_threshold"])
        masked["predicted_center"] = raw["predicted_center"]
        masked["observed_center"] = raw["observed_center"]
        results[name] = masked
        print(f"  {name:12s}: predicted {raw['predicted_center']:.2f} A -> "
              f"snapped {raw['observed_center']:.2f} A "
              f"(delta {raw['observed_center'] - raw['predicted_center']:+.2f} A)")

    ratios = compute_ratios(results, cfg["snr_threshold"])
    df = build_dataframe(flux, results, ratios)

    out_dir = resolve_output_dirs(cfg["output_dir"])
    csv_path = out_dir / cfg["csv_name"]
    df.to_csv(csv_path, index=False)
    print(f"\nSaved tidy spaxel table -> {csv_path}  ({len(df)} rows)")

    print_eda_summary(df, results, ratios, cfg["snr_threshold"])

    figures_dir = out_dir / cfg["figures_dirname"]
    make_figures(results, ratios, figures_dir)
    print(f"\nSaved figures -> {figures_dir}")

    print_interpretation(ratios)

    return dict(flux=flux, var=var, wave=wave, results=results, ratios=ratios, df=df)


if __name__ == "__main__":
    run_pipeline()
