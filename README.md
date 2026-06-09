# backpack_frequency_sweep_sanity_check

End-to-end acoustic QA pipeline for bird-backpack FM radio transmitters.
A log-sweep is played through a speaker, recorded via the transmitter's
receiver, then compared against the reference to estimate bandwidth.

---

## Python packages

```
pip install -r requirements.txt
```

| Package | Min version | Purpose |
|---|---|---|
| click | 8.0.0 | CLI subcommands |
| numpy | 1.20.0 | Array maths |
| soundfile | 0.11.0 | WAV I/O |
| scipy | 1.7.0 | DSP (Welch, Butterworth, ZNCC) |
| matplotlib | 3.4.0 | Report plots & spectrograms |
| sounddevice | 0.4.5 | Real-time audio playback |

---

## qa_sweep.py

Streamlined single-script QA tool for bird-backpack FM radio transmitters.

### Subcommands

| Command | Description |
|---|---|
| `generate` | Write a log-sweep WAV to disk |
| `play` | Play it through the speaker |
| `analyze` | Compare a received WAV against the sweep and report bandwidth |

### Typical session

```bash
# 1. Generate the reference sweep
python qa_sweep.py generate

# 2. Place backpack on speaker, start recording on receiver
python qa_sweep.py play
# … stop recorder, copy WAV to this machine …

# 3. Analyse the recording
python qa_sweep.py analyze received.wav
```

### Output

`analyze` produces a three-panel PNG (`qa_report.png` by default):

1. **ZNCC curve** — zero-normalised cross-correlation vs lag; peak marks the
   temporal alignment between the playback and the recording.
2. **PSD overlay** — Welch power-spectral-density of the reference sweep
   and the aligned received segment.
3. **Frequency response** — `H(f) = PSD_received / PSD_sweep` with
   −3 dB and −6 dB cut-off markers.

### Options

```
generate
  --out TEXT          Output WAV path          [default: qa_sweep.wav]
  --duration FLOAT    Sweep duration in seconds [default: 20.0]
  --samplerate INT    Sample rate in Hz         [default: 44100]

play
  SWEEP               Reference WAV             [default: qa_sweep.wav]
  --device TEXT       Audio output device name or index
  --loops INT         Number of times to play   [default: 1]

analyze
  RECORDING           Received WAV (required)
  --sweep TEXT        Reference sweep WAV       [default: qa_sweep.wav]
  --out TEXT          Output plot PNG           [default: qa_report.png]
  --max-lag FLOAT     ZNCC search window ±s     [default: 60.0]
```

---

## demo_sweep.py

Generates a **realistically perturbed** version of the reference sweep to
demonstrate and validate the analysis pipeline without physical hardware.

### Perturbations applied

| Perturbation | Default | Purpose |
|---|---|---|
| 6-pole Butterworth LPF | 3 200 Hz | Transmitter bandwidth limit |
| Slow AM (0.5 Hz, ±8 %) | fixed | RF fading |
| Additive white Gaussian noise | SNR 20 dB | Channel noise |
| DC bias | 0.02 | ADC artefact |
| Start-time offset | 1.5 s | Exercises ZNCC alignment |

Every stage **peak-normalizes** its output before the next so SNR figures
remain interpretable and clipping is avoided.

### Usage

```bash
# Quick demo with defaults
python demo_sweep.py

# Custom noise / offset / LPF corner
python demo_sweep.py --snr 15 --offset 3.0 --lp 2000

# Full option list
python demo_sweep.py --help
```

### Output

`demo_report.png` — four-panel figure:

1. **ZNCC curve** (full height, left) — correlation vs lag with the
   maximum clearly marked; demonstrates correct alignment despite the
   artificial offset.
2. **Reference spectrogram** — time × frequency heatmap of the sweep.
3. **Received spectrogram** — same scale; the LPF roll-off and noise
   floor are visually apparent above ~3 kHz.
4. **PSD overlay** (top right) — Welch PSDs of reference and received.
5. **Frequency response** (bottom, full width) — `H(f)` with −3 dB /
   −6 dB cut-off markers.

### Options

```
  --snr FLOAT     SNR of simulated received signal (dB)    [default: 20.0]
  --offset FLOAT  Simulated recording start offset (s)     [default: 1.5]
  --lp FLOAT      LPF cut-off of simulated transmitter Hz  [default: 3200]
  --out TEXT      Output perturbed WAV path                [default: demo_received.wav]
  --report TEXT   Output QA report PNG path                [default: demo_report.png]
  --sweep TEXT    Reference sweep WAV                      [default: qa_sweep.wav]
  --max-lag FLOAT ZNCC ±search window in seconds           [default: 10.0]
```

---

## Signal-processing notes

### Log sweep

The reference is a Legendre/Farina exponential sine sweep:

```
φ(t) = 2π · f_min · L · (e^(t/L) − 1)
L = T / ln(f_max / f_min)
```

covering 20 Hz → 8 kHz in 20 s with 20 ms cosine fade-in/out.

### ZNCC alignment

Two-stage sliding zero-normalised cross-correlation:

1. **Coarse pass** — decimated ×8, searches ±`max_lag` s.
2. **Fine pass** — full resolution, ±3 s around coarse peak.

Each pass normalizes the template to zero-mean / unit-norm and divides
the FFT cross-correlation by the local standard deviation of the recording
window, giving a true Pearson-correlation coefficient in [−1, 1].

### Frequency response

```
H(f) = PSD_received(f) / PSD_sweep(f)
```

Computed via Welch's method (nperseg = 4096).  Only bins where the sweep
PSD is within 40 dB of its peak and f ≥ 30 Hz are used (mask avoids
noise-floor artefacts).  Cut-offs are detected by scanning left from the
passband peak until a sustained roll-off exceeds the threshold.