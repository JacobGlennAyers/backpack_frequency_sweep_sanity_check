#!/usr/bin/env python3
"""
qa_sweep.py
-----------
Streamlined single-script QA tool for bird-backpack FM radio transmitters.

Three subcommands:

  generate   — write a log-sweep WAV to disk
  play       — play it through the speaker
  analyze    — compare a received WAV against the sweep and report bandwidth

Typical session:

  python qa_sweep.py generate
  python qa_sweep.py play
  # ... place backpack on speaker, start recording on receiver, run play,
  #     stop recorder, copy WAV to this machine ...
  python qa_sweep.py analyze received.wav

analyze produces TWO output PNGs:
  qa_report.png       — ZNCC curve · PSD overlay · frequency response
  qa_spectrograms.png — dB-mag spectrogram triptych:
                          Reference | Received | Difference (received − ref)
"""

import json, os, sys, time
import click
import numpy as np
import soundfile as sf
import scipy.signal as ss
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── constants ────────────────────────────────────────────────────────────────

SWEEP_FILE        = "qa_sweep.wav"
REPORT_FILE       = "qa_report.png"
SPECTROGRAM_FILE  = "qa_spectrograms.png"
DEFAULT_FS        = 44100
DEFAULT_DURATION  = 20.0        # seconds
F_MIN, F_MAX      = 20.0, 8000.0
TARGET_AMP        = 0.8

# Spectrogram parameters (shared between both scripts)
SG_NPERSEG  = 1024
SG_NOVERLAP = SG_NPERSEG * 3 // 4
SG_FLOOR_DB = -80.0   # clip floor for display


# ── DSP ──────────────────────────────────────────────────────────────────────

def _normalize(x: np.ndarray) -> np.ndarray:
    """Peak-normalize to ±1, safe against silent signals."""
    peak = np.abs(x).max()
    return x / peak if peak > 1e-9 else x


def make_log_sweep(fs: int, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    L = duration / np.log(F_MAX / F_MIN)
    phase = 2 * np.pi * F_MIN * L * (np.exp(t / L) - 1)
    sig = TARGET_AMP * np.sin(phase)
    fade = int(0.02 * fs)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))
    sig[:fade]  *= ramp
    sig[-fade:] *= ramp[::-1]
    return sig.astype(np.float32)


def load_mono(path: str, target_fs: int | None = None) -> tuple[np.ndarray, int]:
    data, fs = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if target_fs and fs != target_fs:
        mono = ss.resample(mono, int(len(mono) * target_fs / fs)).astype(np.float32)
        fs = target_fs
    return mono, fs


def sliding_zncc(template: np.ndarray, recording: np.ndarray,
                 max_lag: int) -> tuple[int, float, np.ndarray, np.ndarray]:
    """
    Two-stage sliding-window ZNCC with proper per-window local normalisation.

    Stage 1: decimated coarse search over the full ±max_lag window.
    Stage 2: full-resolution refinement around the coarse peak.

    The template is zero-mean / unit-norm normalized before each pass so the
    resulting coefficient is a true Pearson correlation in [−1, 1].
    """
    def _zncc(tmpl, rec, max_lag):
        L = len(tmpl)
        t = (tmpl - tmpl.mean()).astype(np.float64)
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-10:
            raise ValueError("Template is silent.")
        t /= t_norm
        r = rec.astype(np.float64)

        # Numerator via FFT cross-correlation
        full_corr = ss.fftconvolve(r, t[::-1], mode="full")
        centre = L - 1
        lo = max(0, centre - max_lag)
        hi = min(len(full_corr), centre + max_lag + 1)
        lags = np.arange(lo, hi) - centre
        corr = full_corr[lo:hi]

        # Local σ of recording windows via cumsum
        r_pad = np.pad(r, (L - 1, L - 1))
        cs    = np.concatenate([[0.0], np.cumsum(r_pad)])
        cs_sq = np.concatenate([[0.0], np.cumsum(r_pad ** 2)])
        starts = lags + (L - 1)
        valid = (starts >= 0) & (starts + L <= len(r_pad))
        s1 = np.clip(starts + 1,     0, len(cs) - 1)
        e1 = np.clip(starts + L + 1, 0, len(cs) - 1)
        s    = np.where(valid, cs[e1]    - cs[s1 - 1],    0.0)
        s_sq = np.where(valid, cs_sq[e1] - cs_sq[s1 - 1], 1.0)
        mean_r = s / L
        std_r  = np.sqrt(np.maximum(s_sq / L - mean_r ** 2, 0.0))
        std_r  = np.maximum(std_r, 1e-12)

        denom = std_r * np.sqrt(L)
        zncc  = np.clip(corr / denom, -1.0, 1.0)
        peak  = int(np.argmax(np.abs(zncc)))
        return int(lags[peak]), float(zncc[peak]), zncc.astype(np.float32), lags

    # Coarse pass (decimated ×8)
    d = 8
    coarse_lag_d, _, _, _ = _zncc(
        ss.decimate(template.astype(np.float64), d, zero_phase=True).astype(np.float32),
        ss.decimate(recording.astype(np.float64), d, zero_phase=True).astype(np.float32),
        max_lag // d,
    )
    coarse_lag = coarse_lag_d * d

    # Fine pass around coarse peak
    refine = int(3.0 * DEFAULT_FS)
    return _zncc(template, recording, abs(coarse_lag) + refine)


def frequency_response(template: np.ndarray, received: np.ndarray,
                       fs: int, nperseg: int = 4096):
    """
    H(f) = PSD_received / PSD_template via Welch's method.

    Both signals are peak-normalized before PSD estimation.  Only bins where
    the template PSD is within 40 dB of its peak and f ≥ 30 Hz are used.
    """
    f, pt = ss.welch(_normalize(template).astype(np.float64), fs=fs, nperseg=nperseg)
    _, pr = ss.welch(_normalize(received).astype(np.float64), fs=fs, nperseg=nperseg)
    pt_db = 10 * np.log10(np.maximum(pt, 1e-30))
    mask  = (pt_db >= pt_db.max() - 40) & (f >= 30)
    resp  = np.full_like(f, np.nan)
    resp[mask] = 10 * np.log10(
        np.maximum(pr[mask], 1e-30) / np.maximum(pt[mask], 1e-30))
    return f, resp


def find_cutoff(freqs, response_db, rolloff_db, f_min=30.0):
    """
    First frequency above the passband peak where the response has rolled off
    by rolloff_db and stays below for at least 3 consecutive bins.
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


# ── spectrogram helpers ───────────────────────────────────────────────────────

def _compute_spectrogram_db(sig: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a peak-normalized dB-magnitude spectrogram.

    Returns (freqs_hz, times_s, Sxx_db) where Sxx_db is clipped to
    [SG_FLOOR_DB, 0] dB so every spectrogram shares the same reference.
    """
    sig_n = _normalize(sig).astype(np.float64)
    f, t, Sxx = ss.spectrogram(sig_n, fs=fs,
                                nperseg=SG_NPERSEG,
                                noverlap=SG_NOVERLAP,
                                window="hann")
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-20))
    # Peak-normalize so ref = 0 dB at each spectrogram's loudest bin
    Sxx_db -= Sxx_db.max()
    Sxx_db  = np.clip(Sxx_db, SG_FLOOR_DB, 0.0)
    return f, t, Sxx_db


def plot_spectrogram_triptych(
        tmpl: np.ndarray,
        received_seg: np.ndarray,
        fs: int,
        title: str,
        out: str,
        f_max_khz: float = F_MAX / 1000,
        c3: float | None = None,
        c6: float | None = None,
):
    """
    Three-panel dB-magnitude spectrogram figure saved to `out`:

      Panel 1 — Reference sweep (Blues)
      Panel 2 — Received / aligned segment (Greens)
      Panel 3 — Difference: received − reference (RdBu_r, diverging)

    All three share the same frequency axis and time axis.  Panels 1 & 2
    share the same colour scale [SG_FLOOR_DB, 0] dB.  Panel 3 uses a
    symmetric ±20 dB diverging scale so gains and losses are immediately
    visible.  Horizontal dashed lines mark the −3 dB / −6 dB cut-off
    frequencies when supplied.
    """
    f_ref, t_ref, Sxx_ref = _compute_spectrogram_db(tmpl, fs)
    f_rec, t_rec, Sxx_rec = _compute_spectrogram_db(received_seg, fs)

    # ── Difference: interpolate onto ref time/freq grid if needed, then subtract
    # Both signals are the same length after alignment, so grids match exactly.
    Sxx_diff = Sxx_rec - Sxx_ref   # positive = gain, negative = loss

    f_khz = f_ref / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True,
                             layout="constrained")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    vmin_abs, vmax_abs = SG_FLOOR_DB, 0.0
    diff_lim = 20.0   # ±20 dB symmetric scale for the difference panel

    panels = [
        (axes[0], Sxx_ref, t_ref, "Blues",   vmin_abs, vmax_abs,
         "Reference sweep", "dB (normalized)"),
        (axes[1], Sxx_rec, t_rec, "Greens",  vmin_abs, vmax_abs,
         "Received (aligned)", "dB (normalized)"),
        (axes[2], Sxx_diff, t_ref, "RdBu_r", -diff_lim, diff_lim,
         "Difference  (received − reference)", "ΔdB"),
    ]

    cutoff_colors = {"−3 dB": ("#ca8a04", c3), "−6 dB": ("#dc2626", c6)}

    for ax, Sxx, t_ax, cmap, vmin, vmax, panel_title, cb_label in panels:
        im = ax.pcolormesh(
            t_ax, f_khz, Sxx,
            vmin=vmin, vmax=vmax,
            cmap=cmap, shading="gouraud", rasterized=True,
        )
        ax.set_ylim(0, f_max_khz * 1.02)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Frequency (kHz)", fontsize=9)
        ax.set_title(panel_title, fontsize=10, pad=4)

        cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=28)
        cb.set_label(cb_label, fontsize=8)
        if vmin == -diff_lim:
            cb.set_ticks([-diff_lim, -10, 0, 10, diff_lim])

        # Cut-off frequency markers
        for label, (color, cf) in cutoff_colors.items():
            if cf is not None:
                ax.axhline(cf / 1000, color=color, lw=1.0, ls="--",
                           label=f"{label}: {cf:.0f} Hz")
        if any(v is not None for _, v in cutoff_colors.values()):
            ax.legend(fontsize=7, loc="upper right",
                      framealpha=0.75, handlelength=1.2)

    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── subcommands ──────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Bird backpack FM transmitter QA — log sweep edition."""


@cli.command()
@click.option("--out",        default=SWEEP_FILE,      show_default=True,
              help="Output WAV path.")
@click.option("--duration",   default=DEFAULT_DURATION, type=float, show_default=True,
              help="Sweep duration in seconds.")
@click.option("--samplerate", default=DEFAULT_FS,       type=int,   show_default=True,
              help="Sample rate in Hz.")
def generate(out, duration, samplerate):
    """Generate the log-sweep reference WAV."""
    sweep = make_log_sweep(samplerate, duration)
    sf.write(out, sweep, samplerate, subtype="PCM_16")
    click.echo(f"Wrote {out}  ({duration:.0f} s, {F_MIN:.0f}–{F_MAX:.0f} Hz log sweep)")


@cli.command()
@click.argument("sweep", default=SWEEP_FILE)
@click.option("--device", default=None,
              help="Audio output device name or index (see "
                   "'python -c \"import sounddevice; print(sounddevice.query_devices())\"').")
@click.option("--loops", default=1, type=int, show_default=True,
              help="Number of times to play the sweep.")
def play(sweep, device, loops):
    """Play the sweep WAV through the speaker.

    SWEEP defaults to qa_sweep.wav.  Start the SD-card recorder before
    running this command.
    """
    try:
        import sounddevice as sd
        sd.query_devices()
    except Exception:
        click.echo("ERROR: sounddevice / PortAudio not available.\n"
                   "Play the WAV manually with any audio player.", err=True)
        sys.exit(1)

    if not os.path.exists(sweep):
        click.echo(f"ERROR: '{sweep}' not found. Run 'generate' first.", err=True)
        sys.exit(1)

    audio, fs = load_mono(sweep)
    kwargs = {"samplerate": fs}
    if device is not None:
        kwargs["device"] = int(device) if str(device).isdigit() else device

    for i in range(loops):
        if loops > 1:
            click.echo(f"Loop {i+1}/{loops}")
        click.echo(f"Playing {sweep} ({len(audio)/fs:.0f} s) — Ctrl+C to abort")
        sd.play(audio, **kwargs)
        sd.wait()

    click.echo("Done. Stop the recorder and copy the WAV to this machine.")
    click.echo(f"Then run:  python qa_sweep.py analyze <received.wav>")


@cli.command()
@click.argument("recording")
@click.option("--sweep",   default=SWEEP_FILE,     show_default=True,
              help="Reference sweep WAV.")
@click.option("--out",     default=REPORT_FILE,    show_default=True,
              help="Output report PNG (ZNCC + PSD + freq response).")
@click.option("--sg-out",  default=SPECTROGRAM_FILE, show_default=True,
              help="Output spectrogram triptych PNG.")
@click.option("--max-lag", default=60.0, type=float, show_default=True,
              help="ZNCC search window ±seconds.")
def analyze(recording, sweep, out, sg_out, max_lag):
    """Compare RECORDING against the reference sweep and report bandwidth.

    Produces two PNGs:

    \b
    <--out>    3-panel report: ZNCC curve · PSD overlay · freq response
    <--sg-out> 3-panel spectrogram triptych:
                 Reference dB mag | Received dB mag | Difference (received−ref)
    """
    if not os.path.exists(sweep):
        click.echo(f"ERROR: sweep file '{sweep}' not found. Run 'generate' first.",
                   err=True)
        sys.exit(1)

    tmpl, fs = load_mono(sweep)
    rec,  _  = load_mono(recording, target_fs=fs)

    click.echo(f"Sweep    : {len(tmpl)/fs:.0f} s @ {fs} Hz")
    click.echo(f"Recording: {len(rec)/fs:.0f} s @ {fs} Hz")
    click.echo(f"Searching (±{max_lag:.0f} s)…", nl=False)

    best_lag, peak_zncc, zncc_curve, lags = sliding_zncc(
        tmpl, rec, int(max_lag * fs))

    click.echo(f"  ZNCC = {peak_zncc:.3f}  @ {best_lag/fs:+.2f} s")
    if peak_zncc < 0.3:
        click.echo("⚠  Low ZNCC — check the recording covers the full playback.")

    start = best_lag
    received_seg = rec[start : start + len(tmpl)]
    if len(received_seg) < len(tmpl):
        received_seg = np.pad(received_seg, (0, len(tmpl) - len(received_seg)))

    freqs, resp = frequency_response(tmpl, received_seg, fs)
    c3 = find_cutoff(freqs, resp, 3.0)
    c6 = find_cutoff(freqs, resp, 6.0)

    click.echo(f"−3 dB cut-off : {f'{c3:.0f} Hz' if c3 else 'not detected'}")
    click.echo(f"−6 dB cut-off : {f'{c6:.0f} Hz' if c6 else 'not detected'}")

    # ── Report PNG (ZNCC + PSD + freq response) ───────────────────────────────
    _plot_report(tmpl, received_seg, fs,
                 zncc_curve, lags, best_lag, peak_zncc,
                 freqs, resp, c3, c6,
                 recording_name=recording,
                 out=out)
    click.echo(f"Report       → {out}")

    # ── Spectrogram triptych PNG ──────────────────────────────────────────────
    sg_title = (
        f"Spectrogram comparison — {os.path.basename(recording)}"
        + (f"   −3 dB: {c3:.0f} Hz" if c3 else "")
        + (f"   −6 dB: {c6:.0f} Hz" if c6 else "")
    )
    plot_spectrogram_triptych(tmpl, received_seg, fs,
                              title=sg_title,
                              out=sg_out,
                              c3=c3, c6=c6)
    click.echo(f"Spectrograms → {sg_out}")


# ── internal plot helper (report) ─────────────────────────────────────────────

def _plot_report(tmpl, received_seg, fs,
                 zncc_curve, lags, best_lag, peak_zncc,
                 freqs, resp, c3, c6,
                 recording_name: str, out: str):
    """Three-panel QA report: ZNCC · PSD overlay · frequency response."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Backpack QA — {os.path.basename(recording_name)}"
        + (f"   −3 dB: {c3:.0f} Hz" if c3 else "")
        + (f"   −6 dB: {c6:.0f} Hz" if c6 else ""),
        fontsize=11, fontweight="bold",
    )

    # Panel 1: ZNCC
    ax = axes[0]
    lag_s = lags / fs
    ax.plot(lag_s, zncc_curve, color="#7c3aed", lw=0.8)
    ax.axvline(best_lag / fs, color="red", lw=1.2, ls="--",
               label=f"peak {peak_zncc:.3f} @ {best_lag/fs:.1f}s")
    ax.axhline(0,   color="gray",    lw=0.5)
    ax.axhline(0.3, color="#ca8a04", lw=0.8, ls=":", label="0.3 threshold")
    ax.set_xlabel("Lag (s)"); ax.set_ylabel("ZNCC")
    ax.set_title("Cross-correlation"); ax.legend(fontsize=8)

    # Panel 2: PSDs
    ax = axes[1]
    f_t, p_t = ss.welch(_normalize(tmpl).astype(np.float64),         fs=fs, nperseg=4096)
    f_r, p_r = ss.welch(_normalize(received_seg).astype(np.float64), fs=fs, nperseg=4096)
    ax.semilogy(f_t, p_t, color="#2563eb", lw=0.9, label="Sweep (ref)")
    ax.semilogy(f_r, p_r, color="#dc2626", lw=0.9, label="Received")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title("Power spectral density")
    ax.set_xlim(0, fs / 2); ax.legend(fontsize=8)

    # Panel 3: frequency response
    ax = axes[2]
    valid = ~np.isnan(resp)
    ax.plot(freqs[valid], resp[valid], color="#16a34a", lw=1.2)
    ax.axhline(0,  color="gray",    lw=0.6, ls="--")
    ax.axhline(-3, color="#ca8a04", lw=0.8, ls="--", label="−3 dB")
    ax.axhline(-6, color="#dc2626", lw=0.8, ls="--", label="−6 dB")
    if c3: ax.axvline(c3, color="#ca8a04", lw=1.0, ls=":",
                      label=f"−3 dB: {c3:.0f} Hz")
    if c6: ax.axvline(c6, color="#dc2626", lw=1.0, ls=":",
                      label=f"−6 dB: {c6:.0f} Hz")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Level (dB)")
    ax.set_title("Frequency response"); ax.set_ylim(-40, 10)
    ax.set_xlim(0, fs / 2); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cli()