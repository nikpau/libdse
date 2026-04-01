# Speech Enhancement via Denoising Autoencoder

A PyTorch implementation of a **Denoising Autoencoder (DAE)** for single-channel speech enhancement, following the approach of Lu et al. (2012) — *Speech Enhancement Based on Deep Denoising Autoencoder*.

---

## The Idea

Real-world speech is almost never clean.  Recordings captured in cars, open offices, or outdoor environments are corrupted by additive noise — fan hum, traffic, background chatter — that degrades both intelligibility and downstream processing (ASR, speaker verification, etc.).

A **denoising autoencoder** addresses this by learning a mapping from *noisy* spectral features back to *clean* spectral features.  During training the model sees pairs `(noisy_mel, clean_mel)` and is penalised for any reconstruction error.  At inference time only the noisy side is available; the encoder compresses it into a latent representation, and the decoder projects that representation back to a clean estimate.

```
Noisy speech                              Clean speech estimate
    │                                            ▲
    ▼                                            │
┌───────────┐     latent z     ┌────────────┐    │
│  Encoder  │ ───────────────► │   Decoder  │ ───┘
└───────────┘                  └────────────┘
```

Working in the **mel-filterbank** domain rather than raw waveforms brings two advantages:

1. **Perceptual alignment** — mel spacing mirrors how the human auditory system resolves pitch, so the model optimises a representation that is directly meaningful for speech quality.
2. **Dimensionality reduction** — a 40- or 80-bin mel spectrum is far smaller than a full FFT, which makes training faster and less prone to over-fitting.

The plan is to synthesise noisy training data by mixing LibriSpeech utterances with environmental noise corpora (e.g. MUSAN, DEMAND), pre-process both the clean and noisy streams through the same feature pipeline, and train the DAE to reconstruct the clean features from the noisy ones.

---

## Repository Structure

```
speech_enhancement/
├── src/
│   └── dae/
│       ├── __init__.py          # Package init
│       └── data/
│           └── dataprep.py      # Feature extraction & dataset class
├── data/
│   └── train-clean-100/         # LibriSpeech train-clean-100 corpus (not included in repo)
├── tests/
│   ├── resources/               # Small FLAC fixtures for unit tests
│   └── test_dataset.py          # Integration tests for the dataset pipeline
└── pyproject.toml
```

---

## What We Have So Far

### Feature Extraction Pipeline (`src/dae/data/dataprep.py`)

The pre-processing pipeline converts raw LibriSpeech FLAC files into ready-to-train NumPy arrays:

```
FLAC file
   │
   ▼  librosa.load (16 kHz, mono)
PCM waveform
   │
   ▼  STFT  (Hann window, configurable chunksize & overlap)
Complex spectrogram  (n_fft/2 + 1, n_frames)
   │
   ▼  Mel filterbank projection
Log-mel power spectrogram  (n_mels, n_frames)
   │
   ▼  Sliding window of width chunks_per_feature
Individual samples  (n_mels, chunks_per_feature)
   │
   ▼  Stacked & saved as UUID-named .npy shards
Preprocessed archive  (entry_point/preprocessed/*.npy)
```

An MD5 checksum is recorded for every shard in a `.preproc` manifest at the dataset root, allowing re-use of pre-processed data across runs without re-computation.

Processing can be run **single-threaded** or **multi-processed**:

```python
from pathlib import Path
from dae.data.dataprep import LibriSpeechDataset

ds = LibriSpeechDataset(Path("data/train-clean-100"))
ds.prepare(
    n_cpu=8,          # worker processes (1 = single-threaded)
    n_mels=80,        # mel filterbank bins
    chunksize=25,     # STFT window length [ms]
    overlap=10,       # STFT hop length [ms]
    chunks_per_feature=20,  # time frames per training sample
)

# ds is a PyTorch IterableDataset — plug straight into DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=64)
```

Each item yielded is a `torch.Tensor` of shape `(n_mels, chunks_per_feature)`.

> **Note on multi-processing:** Each worker process is initialised with all BLAS/OpenMP thread-count environment variables set to `1` to prevent thread oversubscription.  With `n_cpu` workers you get exactly `n_cpu` active threads, which gives linear scaling up to the physical core count.

### Key Classes

| Class | Role |
|---|---|
| `SampleWarehouse` | Splits FLAC paths into ≤128 shards, runs the mel feature extraction, writes `.npy` binaries and `.preproc` manifest |
| `LibriSpeechDataset` | `torch.utils.data.IterableDataset` wrapper; validates the corpus root, owns a `SampleWarehouse`, guards iteration behind a `prepare()` call |

---

## Installation

Requires Python ≥ 3.12.

```bash
# Clone the repository
git clone <repo-url>
cd speech_enhancement

# Install in editable mode (uv recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Development Dependencies

```bash
uv pip install -e ".[dev]"
```

---

## Running Tests

```bash
uv run pytest
```

The test suite covers:

- FLAC → WAV conversion (happy path, missing source, corrupt file)
- `SampleWarehouse` initialisation, sharding, mel-bin counts, array shapes, checksum recording
- `LibriSpeechDataset` entry-point validation, iterator guards, tensor shapes
- Single-process vs. multi-process output equivalence

---

## Data

Download [LibriSpeech train-clean-100](https://www.openslr.org/12) (~6.3 GB) and extract it so the directory structure is:

```
data/
└── train-clean-100/
    └── LibriSpeech/
        └── train-clean-100/
            └── <speaker>/
                └── <chapter>/
                    └── *.flac
```

Pass the `data/train-clean-100` directory as `entry_point` to `LibriSpeechDataset`.

---

## Roadmap

- [ ] Synthesise noisy training pairs (mix LibriSpeech + noise corpus)
- [ ] Implement the DAE encoder–decoder architecture (I will try a VAE)
- [ ] Training loop with reconstruction loss (MSE on mel features)
- [ ] Evaluation metrics: PESQ, STOI, SI-SDR on a held-out test set
- [ ] Griffin-Lim / vocoder reconstruction to recover waveform from enhanced mel
