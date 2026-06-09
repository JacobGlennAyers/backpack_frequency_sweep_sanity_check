#!/usr/bin/env python3
"""
demo_sweep.py
-------------
Demonstration script for qa_sweep.py that generates a realistically perturbed
version of the reference log-sweep, simulating what a bird-backpack FM
transmitter might actually record.

Perturbations applied
---------------------
  * Additive white Gaussian noise (SNR ~ 20 dB)
  * Random start-time offset (0–3 s) so ZNCC alignment is non-trivial
  * Gentle low-pass roll-off above ~3 kHz (6-pole Butterworth) mimicking the
    transmitter's high-frequency bandwidth limit
  * Slight amplitude modulation (0.5 Hz, ±8 %) simulating RF fading
  * DC bias of 0.02 (common in cheap ADCs)

After generating the perturbed WAV the script calls the three qa_sweep
subcommands in sequence (generate → analyze) and saves the report PNG.

Usage
-----
  python demo_sweep.py [--snr DB] [--offset S] [--out PERTURBED_WAV]
"""

import os
import sys
import click
import numpy as np
import soundfile as sf
import scipy.signal as ss
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── shared constants (keep in sync with qa_sweep.py) ─────────────────────────
SWEEP_FILE        = "qa_sweep.wav"
DEMO_RECEIVED     = "demo_received.wav"
DEMO_REPORT       = "demo_report.png"
DEMO_SG_REPORT    = "demo_spectrograms.png"
DEFAULT_FS        = 44100
DEFAULT_DURATION  = 20.0
F_MIN, F_MAX      = 20.0, 8000.0
TARGET_AMP        = 0.8


# ── DSP helpers (self-contained copies so the demo has no import dependency) ──

def _make_log_sweep(fs: int, duration: float) -> np.ndarray:
    t     = np.linspace(0, duration, int(duration * fs), endpoint=False)
    L     = duration / np.log(F_MAX / F_MIN)
    phase = 2 * np.pi * F_MIN * L * (np.exp(t / L) - 1)
    sig   = TARGET_AMP * np.sin(phase)
    fade  = int(0.02 * fs)
    ramp  = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))
    sig[:fade]  *= ramp
    sig[-fade:] *= ramp[::-1]
    return sig.astype(np.float32)


def _normalize(x: np.ndarray) -> np.ndarray:
    """Peak-normalize to ±1, safe against silent signals."""
    peak = np.abs(x).max()
    return x / peak if peak > 1e-9 else x


def _perturb(sweep: np.ndarray, fs: int,
             snr_db: float = 20.0,
             offset_samples: int = 0,
             lp_cutoff_hz: float = 3200.0,
             am_depth: float = 0.08,
             am_freq_hz: float = 0.5,
             dc_bias: float = 0.02) -> np.ndarray:
    """
    Apply a chain of realistic perturbations to `sweep`.

    Each intermediate signal is peak-normalized before the next stage so
    that SNR figures remain interpretable.

    Parameters
    ----------
    sweep           : reference sweep (float32, peak ~ TARGET_AMP)
    fs              : sample rate
    snr_db          : additive noise SNR in dB (relative to the sweep peak)
    offset_samples  : prepend this many silent samples (simulates record-start delay)
    lp_cutoff_hz    : -3 dB corner of the simulated transmitter LPF
    am_depth        : fractional amplitude-modulation depth (0–1)
    am_freq_hz      : AM envelope rate in Hz
    dc_bias         : constant DC offset added by the simulated ADC
    """
    rng = np.random.default_rng(seed=42)   # reproducible demo

    # 1. Low-pass roll-off  ── normalize input first
    s = _normalize(sweep.copy().astype(np.float64))
    sos = ss.butter(6, lp_cutoff_hz / (fs / 2), btype="low", output="sos")
    s   = ss.sosfiltfilt(sos, s)
    s   = _normalize(s)

    # 2. Amplitude modulation (slow RF fading)
    t   = np.arange(len(s)) / fs
    env = 1.0 + am_depth * np.sin(2 * np.pi * am_freq_hz * t)
    s  *= env
    s   = _normalize(s)

    # 3. Additive white Gaussian noise
    sig_power  = np.mean(s ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=len(s))
    s    += noise
    s     = _normalize(s)

    # 4. DC bias
    s += dc_bias

    # 5. Time offset (prepend silence)
    if offset_samples > 0:
        s = np.concatenate([np.zeros(offset_samples, dtype=s.dtype), s])

    return s.astype(np.float32)


def _load_mono(path: str, target_fs: int | None = None) -> tuple[np.ndarray, int]:
    data, fs = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if target_fs and fs != target_fs:
        mono = ss.resample(mono, int(len(mono) * target_fs / fs)).astype(np.float32)
        fs   = target_fs
    return mono, fs


def _zncc_align(template: np.ndarray, recording: np.ndarray,
                max_lag: int) -> tuple[int, float, np.ndarray, np.ndarray]:
    """
    Two-stage ZNCC: decimated coarse pass then full-resolution fine pass.

    Both template and recording are zero-mean / unit-std normalized before
    each correlation pass so the ZNCC coefficient is always in [-1, 1].
    """
    def _run(tmpl: np.ndarray, rec: np.ndarray, max_lag: int):
        L    = len(tmpl)
        # ── per-pass normalization ──────────────────────────────────────────
        t    = (tmpl - tmpl.mean()).astype(np.float64)
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-10:
            raise ValueError("Template is silent after normalization.")
        t   /= t_norm
        r    = rec.astype(np.float64)

        # Numerator: FFT cross-correlation
        full_corr = ss.fftconvolve(r, t[::-1], mode="full")
        centre    = L - 1
        lo  = max(0, centre - max_lag)
        hi  = min(len(full_corr), centre + max_lag + 1)
        lags = np.arange(lo, hi) - centre
        corr = full_corr[lo:hi]

        # Denominator: local σ of recording via cumsum (normalized by √L so
        # the ZNCC is equivalent to the Pearson correlation coefficient)
        r_pad  = np.pad(r, (L - 1, L - 1))
        cs     = np.concatenate([[0.0], np.cumsum(r_pad)])
        cs_sq  = np.concatenate([[0.0], np.cumsum(r_pad ** 2)])
        starts = lags + (L - 1)
        valid  = (starts >= 0) & (starts + L <= len(r_pad))
        s1 = np.clip(starts + 1,     0, len(cs) - 1)
        e1 = np.clip(starts + L + 1, 0, len(cs) - 1)
        s_sum    = np.where(valid, cs[e1]    - cs[s1 - 1],    0.0)
        s_sq_sum = np.where(valid, cs_sq[e1] - cs_sq[s1 - 1], 1.0)
        mean_r   = s_sum / L
        var_r    = np.maximum(s_sq_sum / L - mean_r ** 2, 0.0)
        std_r    = np.maximum(np.sqrt(var_r), 1e-12)

        # ZNCC = numerator / (||t|| · local_std(r) · √L)
        #      = corr /  (1   · std_r · √L)   (||t|| already divided out)
        denom = std_r * np.sqrt(L)
        zncc  = np.clip(corr / denom, -1.0, 1.0)
        peak  = int(np.argmax(np.abs(zncc)))
        return int(lags[peak]), float(zncc[peak]), zncc.astype(np.float32), lags

    # Stage 1: coarse (decimated ×8)
    d = 8
    tmpl_d = ss.decimate(template.astype(np.float64), d, zero_phase=True).astype(np.float32)
    rec_d  = ss.decimate(recording.astype(np.float64), d, zero_phase=True).astype(np.float32)
    coarse_lag_d, _, _, _ = _run(tmpl_d, rec_d, max_lag // d)
    coarse_lag = coarse_lag_d * d

    # Stage 2: fine (±3 s around coarse peak)
    refine = int(3.0 * DEFAULT_FS)
    return _run(template, recording, abs(coarse_lag) + refine)


def _frequency_response(template: np.ndarray, received: np.ndarray,
                        fs: int, nperseg: int = 4096):
    """
    Estimate H(f) = PSD_received / PSD_template.

    Both PSDs are computed via Welch's method.  Only bins where the template
    PSD is within 40 dB of its peak (and f ≥ 30 Hz) are considered valid to
    avoid noise-floor artefacts.
    """
    f, pt = ss.welch(_normalize(template).astype(np.float64),  fs=fs, nperseg=nperseg)
    _, pr = ss.welch(_normalize(received).astype(np.float64),  fs=fs, nperseg=nperseg)
    pt_db = 10 * np.log10(np.maximum(pt, 1e-30))
    mask  = (pt_db >= pt_db.max() - 40) & (f >= 30)
    resp  = np.full_like(f, np.nan)
    resp[mask] = 10 * np.log10(
        np.maximum(pr[mask], 1e-30) / np.maximum(pt[mask], 1e-30))
    return f, resp


def _find_cutoff(freqs, response_db, rolloff_db, f_min=30.0):
    """
    Find the first frequency above the passband peak where the response
    has rolled off by `rolloff_db` dB and stays there for at least 3 bins.
    Scans rightward (toward high frequencies) from the peak.
    """
    valid = (~np.isnan(response_db)) & (freqs >= f_min)
    if valid.sum() < 8:
        return None
    fv = freqs[valid]
    rv = response_db[valid].astype(np.float64)
    w  = max(3, len(rv) // 100) | 1
    rs = scipy.ndimage.uniform_filter1d(rv, size=w)
    peak_idx = int(np.argmax(rs))
    ref = float(rs[peak_idx])
    run = 0
    for i in range(peak_idx, len(rs)):
        if rs[i] < ref - rolloff_db:
            run += 1
            if run >= 3:
                return float(fv[max(0, i - 2)])
        else:
            run = 0
    return None


# ── import shared spectrogram helper from qa_sweep ────────────────────────────
# We do this at function-call time to avoid a hard circular dependency at
# module load, and fall back gracefully if qa_sweep.py is not on the path.

def _get_triptych_fn():
    import importlib.util, pathlib
    here = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "qa_sweep", here / "qa_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.plot_spectrogram_triptych


# ── main report figure ────────────────────────────────────────────────────────

def _plot(tmpl, received_seg, fs,
          zncc_curve, lags, best_lag, peak_zncc,
          freqs, resp, c3, c6,
          recording_name: str, out: str, sg_out: str):
    """
    Two output figures:

    <out>    — 3-panel report: ZNCC (tall) · PSD overlay · freq response
    <sg_out> — 3-panel spectrogram triptych (via qa_sweep.plot_spectrogram_triptych):
                 Reference dB mag | Received dB mag | Difference (received − ref)
    """
    # ── Figure 1: ZNCC + PSD + freq response ─────────────────────────────────
    fig = plt.figure(figsize=(15, 5), layout="constrained")
    fig.suptitle(
        f"Bird-Backpack FM QA — {os.path.basename(recording_name)}"
        + (f"   −3 dB: {c3:.0f} Hz" if c3 else "")
        + (f"   −6 dB: {c6:.0f} Hz" if c6 else ""),
        fontsize=12, fontweight="bold",
    )

    gs = fig.add_gridspec(1, 3, wspace=0.36)

    # ZNCC (full height left)
    ax_zncc = fig.add_subplot(gs[0])
    lag_s = lags / fs
    ax_zncc.plot(lag_s, zncc_curve, color="#7c3aed", lw=0.9, alpha=0.85,
                 label="ZNCC")
    ax_zncc.axvline(best_lag / fs, color="#dc2626", lw=1.4, ls="--",
                    label=f"peak {peak_zncc:.3f}\n@ {best_lag/fs:+.2f} s")
    ax_zncc.axhline(0,   color="gray",    lw=0.5)
    ax_zncc.axhline(0.3, color="#ca8a04", lw=0.8, ls=":", label="0.3 threshold")
    # shade the single maximum sample
    peak_mask = np.abs(zncc_curve) == np.abs(zncc_curve).max()
    ax_zncc.fill_between(lag_s, zncc_curve, where=peak_mask,
                         color="#dc2626", alpha=0.7)
    ax_zncc.set_xlabel("Lag (s)", fontsize=9)
    ax_zncc.set_ylabel("ZNCC coefficient", fontsize=9)
    ax_zncc.set_title("Zero-Normalised\nCross-Correlation", fontsize=10)
    ax_zncc.legend(fontsize=8, loc="upper left")
    ax_zncc.set_ylim(-1.05, 1.05)

    # PSD overlay
    ax_psd = fig.add_subplot(gs[1])
    f_t, p_t = ss.welch(_normalize(tmpl).astype(np.float64),         fs=fs, nperseg=4096)
    f_r, p_r = ss.welch(_normalize(received_seg).astype(np.float64), fs=fs, nperseg=4096)
    ax_psd.semilogy(f_t / 1000, p_t, color="#2563eb", lw=0.9, label="Sweep (ref)")
    ax_psd.semilogy(f_r / 1000, p_r, color="#dc2626", lw=0.9, label="Received",
                    alpha=0.8)
    ax_psd.set_xlabel("Frequency (kHz)", fontsize=9)
    ax_psd.set_ylabel("PSD", fontsize=9)
    ax_psd.set_title("Power Spectral Density", fontsize=10)
    ax_psd.set_xlim(0, F_MAX / 1000 * 1.05)
    ax_psd.legend(fontsize=8)

    # Frequency response
    ax_fr = fig.add_subplot(gs[2])
    valid = ~np.isnan(resp)
    ax_fr.plot(freqs[valid] / 1000, resp[valid], color="#16a34a", lw=1.4,
               label="H(f) estimate")
    ax_fr.axhline(0,  color="gray",    lw=0.7, ls="--")
    ax_fr.axhline(-3, color="#ca8a04", lw=0.9, ls="--", label="−3 dB")
    ax_fr.axhline(-6, color="#dc2626", lw=0.9, ls="--", label="−6 dB")
    if c3:
        ax_fr.axvline(c3 / 1000, color="#ca8a04", lw=1.1, ls=":",
                      label=f"−3 dB: {c3:.0f} Hz")
    if c6:
        ax_fr.axvline(c6 / 1000, color="#dc2626", lw=1.1, ls=":",
                      label=f"−6 dB: {c6:.0f} Hz")
    ax_fr.fill_between(freqs[valid] / 1000, resp[valid], -40,
                       where=(resp[valid] > -40),
                       color="#16a34a", alpha=0.08)
    ax_fr.set_xlabel("Frequency (kHz)", fontsize=9)
    ax_fr.set_ylabel("Level (dB)", fontsize=9)
    ax_fr.set_title("Freq. Response  H(f) = PSD_rx / PSD_ref", fontsize=10)
    ax_fr.set_ylim(-40, 10)
    ax_fr.set_xlim(0, F_MAX / 1000 * 1.05)
    ax_fr.legend(fontsize=8)

    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: spectrogram triptych ────────────────────────────────────────
    try:
        triptych = _get_triptych_fn()
    except Exception as e:
        click.echo(f"⚠  Could not import qa_sweep.py for triptych: {e}", err=True)
        return

    sg_title = (
        f"Spectrogram comparison — {os.path.basename(recording_name)}"
        + (f"   −3 dB: {c3:.0f} Hz" if c3 else "")
        + (f"   −6 dB: {c6:.0f} Hz" if c6 else "")
    )
    triptych(tmpl, received_seg, fs,
             title=sg_title, out=sg_out, c3=c3, c6=c6)


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--snr",    default=20.0, show_default=True, type=float,
              help="Signal-to-noise ratio of the simulated received signal (dB).")
@click.option("--offset", default=1.5,  show_default=True, type=float,
              help="Simulated recording start-time offset in seconds (tests ZNCC).")
@click.option("--lp",     default=3200, show_default=True, type=float,
              help="Low-pass cut-off of the simulated transmitter (Hz).")
@click.option("--out",    default=DEMO_RECEIVED, show_default=True,
              help="Output path for the perturbed WAV.")
@click.option("--report", default=DEMO_REPORT,   show_default=True,
              help="Output path for the QA report PNG.")
@click.option("--sg-out", default=DEMO_SG_REPORT, show_default=True,
              help="Output path for the spectrogram triptych PNG.")
@click.option("--sweep",  default=SWEEP_FILE,    show_default=True,
              help="Reference sweep WAV (generate with qa_sweep.py generate).")
@click.option("--max-lag", default=10.0, show_default=True, type=float,
              help="ZNCC ±search window in seconds.")
def main(snr, offset, lp, out, report, sg_out, sweep, max_lag):
    """
    Generate a realistically perturbed sweep and run the full QA analysis.

    \b
    Perturbations applied
    ─────────────────────
      • 6-pole Butterworth LPF at --lp Hz   (transmitter bandwidth limit)
      • Slow amplitude modulation (0.5 Hz, ±8 %)   (RF fading)
      • Additive white Gaussian noise at --snr dB
      • DC bias of 0.02  (ADC artefact)
      • Random start-time offset of --offset s  (exercises ZNCC alignment)
    Every stage peak-normalizes its output so SNR figures are meaningful.

    \b
    Typical use
    ───────────
      python qa_sweep.py generate
      python demo_sweep.py
      # opens demo_report.png
    """
    # ── 1. Load or generate reference sweep ──────────────────────────────────
    if not os.path.exists(sweep):
        click.echo(f"'{sweep}' not found — generating it now …")
        ref = _make_log_sweep(DEFAULT_FS, DEFAULT_DURATION)
        sf.write(sweep, ref, DEFAULT_FS, subtype="PCM_16")
        click.echo(f"  wrote {sweep}")
    else:
        ref, fs_in = _load_mono(sweep)
        if fs_in != DEFAULT_FS:
            click.echo(f"  resampling from {fs_in} → {DEFAULT_FS} Hz")
            ref = ss.resample(ref, int(len(ref) * DEFAULT_FS / fs_in)).astype(np.float32)

    fs = DEFAULT_FS
    click.echo(f"Reference sweep : {len(ref)/fs:.1f} s @ {fs} Hz")

    # ── 2. Perturb ───────────────────────────────────────────────────────────
    offset_samples = int(offset * fs)
    received = _perturb(ref, fs,
                        snr_db=snr,
                        offset_samples=offset_samples,
                        lp_cutoff_hz=lp)
    sf.write(out, received, fs, subtype="PCM_16")
    click.echo(f"Perturbed WAV   : {out}  "
               f"(SNR={snr:.0f} dB, offset={offset:.1f} s, LPF={lp:.0f} Hz)")

    # ── 3. ZNCC alignment ─────────────────────────────────────────────────────
    click.echo(f"ZNCC search (±{max_lag:.0f} s) …", nl=False)
    best_lag, peak_zncc, zncc_curve, lags = _zncc_align(
        ref, received, int(max_lag * fs))
    click.echo(f"  ZNCC = {peak_zncc:.3f}  @ lag {best_lag/fs:+.2f} s  "
               f"(expected ≈ +{offset:.2f} s)")

    if peak_zncc < 0.3:
        click.echo("⚠  Low ZNCC — check the offset or SNR settings.")

    # ── 4. Extract aligned segment ────────────────────────────────────────────
    start = max(0, best_lag)
    seg   = received[start : start + len(ref)]
    if len(seg) < len(ref):
        seg = np.pad(seg, (0, len(ref) - len(seg)))

    # ── 5. Frequency response ─────────────────────────────────────────────────
    freqs, resp = _frequency_response(ref, seg, fs)
    c3 = _find_cutoff(freqs, resp, 3.0)
    c6 = _find_cutoff(freqs, resp, 6.0)
    click.echo(f"−3 dB cut-off   : {f'{c3:.0f} Hz' if c3 else 'not detected'}")
    click.echo(f"−6 dB cut-off   : {f'{c6:.0f} Hz' if c6 else 'not detected'}")

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    _plot(ref, seg, fs,
          zncc_curve, lags, best_lag, peak_zncc,
          freqs, resp, c3, c6,
          recording_name=out,
          out=report,
          sg_out=sg_out)
    click.echo(f"Report          : {report}")
    click.echo(f"Spectrograms    : {sg_out}")


if __name__ == "__main__":
    main()