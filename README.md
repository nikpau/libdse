# Speech Enhancement via Denoising Autoencoder

A PyTorch implementation of a **Denoising Autoencoder (DAE)** for single-channel speech enhancement, following the approach of Lu et al. (2012) - *Speech Enhancement Based on Deep Denoising Autoencoder*.

---

## The Idea

Real-world speech is almost never clean.  Recordings captured in cars, open offices, or outdoor environments are corrupted by additive noise - fan hum, traffic, background chatter - that degrades both intelligibility and downstream processing (ASR, speaker verification, etc.).

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

1. **Perceptual alignment**: mel spacing mirrors how the human auditory system resolves pitch, so the model optimises a representation that is directly meaningful for speech quality.
2. **Dimensionality reduction**: a 40- or 80-bin mel spectrum is far smaller than a full FFT, which makes training faster and less prone to over-fitting.

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

Feature extraction happens **on the fly** during iteration — no pre-processing
or caching step is needed.  For each FLAC file the following pipeline runs:

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
Power spectrogram  (n_mels, n_frames)
   │
   ▼  Sliding window of width chunks_per_feature (non-overlapping)
Flat samples  shape: (n_mels * chunks_per_feature,)
```

Each sample is yielded as a ``(sample, label)`` pair.  Without noise both
elements are the same clean row.  With noise, ``sample`` is the corrupted
version and ``label`` is the clean reference.

```python
from pathlib import Path
from dae.data.dataprep import LibriSpeechDataset

# Clean speech only
ds = LibriSpeechDataset(
    entry_point=Path("data/train-clean-100"),
    n_mels=40,             # mel filterbank bins
    chunksize=16,          # STFT window length [ms]
    overlap=8,             # STFT hop length [ms]
    chunks_per_feature=7,  # frames per sample
)

# ds is a PyTorch IterableDataset — plug straight into DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=64)
for sample, label in loader:
    ...  # sample == label for clean-only mode
```

Each item is a flat `numpy` row of length `n_mels * chunks_per_feature`
(converted to a `torch.Tensor` by the `DataLoader`).

### Key Classes

| Class | Role |
|---|---|
| `LibriSpeechDataset` | `torch.utils.data.IterableDataset` that validates the corpus root and streams mel features from FLAC files on iteration |
| `DEMANDNoiseDataset` | Loads and concatenates DEMAND noise recordings into a single array for on-the-fly noisy speech synthesis |
| `DEMANDNoiseType` | Enum of DEMAND environment names; pass a subset to `LibriSpeechDataset` to select noise types |

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

- `LibriSpeechDataset` entry-point validation and `EntryPointError` handling
- Iterator output: pair shapes, dtype, clean label identity, and noisy-label distinction
- `__repr__` and `__len__` contract

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

- [x] Synthesise noisy training pairs (mix LibriSpeech + DEMAND noise corpus)
- [ ] Implement the DAE encoder–decoder architecture (I will try a VAE)
- [ ] Training loop with reconstruction loss (MSE on mel features)
- [ ] Evaluation metrics: PESQ, STOI, SI-SDR on a held-out test set
- [ ] Griffin-Lim / vocoder reconstruction to recover waveform from enhanced mel
