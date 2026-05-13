"""
Training script for :class:`dae.nets.VanillaAutoEncoder` on the
LibriSpeech train-clean-100 corpus with DEMAND noise augmentation.

The model is trained to reconstruct clean log-mel spectrogram features
from their noisily corrupted counterparts (denoising autoencoder).

Pipeline
--------
* Every batch: compute MSE loss, back-prop, log train loss + gradient norm.
* Every ``VAL_EVERY`` batches: run a quick validation pass over
  ``VAL_BATCHES`` batches of the test set and log val loss, SNR improvement,
  and the ratio of val loss to train loss.
* After each epoch: full validation, model checkpoint if val loss improves.

TensorBoard metrics
-------------------

- **Loss/train** — MSE on the current training mini-batch (smoothed over 100 batches).
- **Loss/val_quick** — MSE on ``VAL_BATCHES`` val batches (logged every ``VAL_EVERY`` train batches).
- **SNR/val_quick** — Signal-to-noise ratio improvement (dB) on the quick val pass: 10·log₁₀(input_noise_power / residual_power).
- **Ratio/val_to_train** — val_loss / recent_train_loss; tracks over-fitting.
- **GradNorm/encoder** — L2 norm of encoder gradients (health check).
- **GradNorm/decoder** — L2 norm of decoder gradients (health check).
- **Loss/val_epoch** — Full val-set MSE at end of each epoch.
- **SNR/val_epoch** — Full val-set SNR improvement at end of each epoch.
"""

import torch
import aese.nets as nets
import aese.data.librispeech as data
from aese.data.features import LogMagnitudeSpectrumExtractor
from aese.data.noise import DEMANDNoiseDataset, DEMANDNoiseType
from pathlib import Path
import time
import math
import textwrap
from typing import NamedTuple
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Hyperparameters(NamedTuple):
    # Model Hyperparameters
    name: str = "simple_autoencoder_logmag_spec_noisy_clean"
    info: str = (
        "Deep Denoising AutoEncoder trained on log magnitude spectrogram features "
        "with noisy -> clean targets, using DEMAND noise augmentation. Follows "
        "architecture (d) from Nossier et al. (2020), 'An Experimental Analysis "
        "of Deep Learning Architectures for Supervised Speech Enhancement' "
        "but with log magnitude spectrogram features instead of power spectogram",
        "no dropout.",
    )
    model_dir: str = "models"

    # Training Hyperparameters
    n_epochs: int = 40
    batch_size: int = 256
    val_every: int = (
        1000  # run a quick val pass every this many training batches
    )
    val_batches: int = 50  # number of val batches used for the quick pass
    log_every: int = 100  # smooth train loss over this many batches

    # Data Hyperparameters
    train_data_path: Path = Path("data/train-clean-100")
    test_data_path: Path = Path("data/test-clean")
    noise_types: list[DEMANDNoiseType] = DEMANDNoiseType.ALL
    DEMAND_entry_point: Path = Path("data/noise/DEMAND")
    window_length: int = 256  # STFT window size in samples (32 ms @ 8 kHz)
    hop_length: int = 128  # STFT hop size in samples (16 ms @ 8 kHz)
    sampling_rate: int = 8_000  # Paper resamples to 8 kHz.

    # Network Structure (from Nossier et al. 2020, "Architecture (d)")
    hidden_layer_struct: list[int] = [2048, 500]
    latent_dim: int = 180
    dropout: float = 0.0

    # LR Scheduler Hyperparameters (ReduceLROnPlateau)
    lr_patience: int = 2  # epochs with no improvement before reducing LR
    lr_factor: float = 0.5  # factor by which LR is reduced
    lr_min: float = 1e-6  # minimum LR


hp = Hyperparameters()


# Guard against import-time side effects
if __name__ == "__main__":
    # Writer to track training with TensorBoard (logs to `<root>/runs`).
    writer = SummaryWriter()

    # Load all requested DEMAND noise environments into a single concatenated array.
    noise_ds = DEMANDNoiseDataset(
        entry_point=hp.DEMAND_entry_point,
        noise_types=hp.noise_types,
        sample_rate=hp.sampling_rate,
    )

    # Both train and test use the same feature settings; only the noise source
    # differs (same noise pool is reused — random offsets make each draw unique).
    _extractor_kwargs = dict(
        sampling_rate=hp.sampling_rate,
        window_length=hp.window_length,
        hop_length=hp.hop_length,
    )
    train_extractor = LogMagnitudeSpectrumExtractor(
        **_extractor_kwargs, noise=noise_ds
    )
    test_extractor = LogMagnitudeSpectrumExtractor(
        **_extractor_kwargs, noise=noise_ds
    )

    librispeech_train = data.LibriSpeechDataset(
        entry_point=hp.train_data_path,
        extractor=train_extractor,
        sample_rate=hp.sampling_rate,
    )

    librispeech_test = data.LibriSpeechDataset(
        entry_point=hp.test_data_path,
        extractor=test_extractor,
        sample_rate=hp.sampling_rate,
    )

    autoencoder = nets.VanillaAutoEncoder(
        input_dim=librispeech_train.sample_shape[0],
        latent_dim=hp.latent_dim,
        hidden_layer_struct=hp.hidden_layer_struct,
        dropout=hp.dropout,
    )
    optimizer = Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=hp.lr_factor,
        patience=hp.lr_patience,
        min_lr=hp.lr_min,
    )
    loss_fn = nn.MSELoss()

    train_data = DataLoader(librispeech_train, batch_size=hp.batch_size)
    test_data = DataLoader(librispeech_test, batch_size=hp.batch_size)

    n_iter = 0
    best_val_loss = torch.inf
    timestamp = time.time()

    def _snr_improvement_db(
        inputs: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Return mean SNR improvement in dB over a batch.

        SNR improvement = 10·log10(input_noise_power / residual_noise_power), where
        *input_noise_power* is the power of ``(inputs - labels)`` and
        *residual_noise_power* is the power of ``(outputs - labels)``.  A positive
        value means the model reduced the noise.

        :param inputs:  Noisy input batch, shape ``(B, F)``.
        :param outputs: Model reconstructions, shape ``(B, F)``.
        :param labels:  Clean reference batch, shape ``(B, F)``.
        :returns: Mean SNR improvement in dB (scalar float).
        """
        input_noise_power = (inputs - labels).pow(2).mean()
        residual_power = (outputs - labels).pow(2).mean()
        # Guard against division by zero or log of zero.
        if residual_power < 1e-12 or input_noise_power < 1e-12:
            return 0.0
        return 10.0 * math.log10((input_noise_power / residual_power).item())

    def _grad_norm(module: nn.Module) -> float:
        """Compute the L2 norm of all gradients in *module*.

        :param module: Any :class:`torch.nn.Module` whose parameters have
            ``.grad`` populated (i.e. after ``loss.backward()``).
        :returns: L2 gradient norm (scalar float), or ``0.0`` if no gradients
            are available yet.
        """
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.detach().pow(2).sum().item()
        return math.sqrt(total)

    def run_quick_val(max_batches: int) -> tuple[float, float]:
        """Run a partial validation pass over *max_batches* batches.

        Temporarily switches the model to eval mode and disables gradient
        computation to save memory, then restores training mode.

        :param max_batches: Maximum number of test batches to evaluate.
        :returns: Tuple of ``(avg_mse_loss, avg_snr_improvement_db)``.
        """
        autoencoder.eval()
        total_loss = 0.0
        total_snr = 0.0
        n_batches = 0

        with torch.no_grad():
            for vinputs, vlabels in test_data:
                if n_batches >= max_batches:
                    break
                vinputs = vinputs.float()
                vlabels = vlabels.float()
                voutputs = autoencoder(vinputs)
                total_loss += loss_fn(voutputs, vlabels).item()
                total_snr += _snr_improvement_db(vinputs, voutputs, vlabels)
                n_batches += 1

        autoencoder.train()
        avg_loss = total_loss / max(n_batches, 1)
        avg_snr = total_snr / max(n_batches, 1)
        return avg_loss, avg_snr

    def train_epoch(
        n_iter: int, epoch_number: int, writer: SummaryWriter
    ) -> tuple[float, int]:
        """Train for one full epoch and return ``(avg_train_loss, n_iter)``.

        Every ``LOG_EVERY`` batches the smoothed training loss is written to
        TensorBoard.  Every ``VAL_EVERY`` batches a quick partial validation is
        run, logging val loss, SNR improvement, val/train loss ratio, and
        gradient norms.

        :param n_iter: Global sample counter at the start of this epoch.
        :param epoch_number: Zero-based epoch index (used for log labels).
        :param writer: Active :class:`~torch.utils.tensorboard.SummaryWriter`.
        :returns: Tuple of ``(recent_smoothed_train_loss, updated_n_iter)``.
        """
        running_loss = 0.0
        recent_loss = 0.0

        for i, (input, label) in enumerate(train_data):
            optimizer.zero_grad()

            prediction = autoencoder(input)
            loss: torch.Tensor = loss_fn(prediction, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_iter += hp.batch_size

            # ── Periodic train logging ────────────────────────────────────────────
            if (i + 1) % hp.log_every == 0:
                recent_loss = running_loss / hp.log_every
                writer.add_scalar("Loss/train", recent_loss, n_iter)
                running_loss = 0.0

            # ── Quick validation every VAL_EVERY batches ──────────────────────────
            if (i + 1) % hp.val_every == 0:
                val_loss, val_snr = run_quick_val(hp.val_batches)
                enc_gnorm = _grad_norm(autoencoder.encoder)
                dec_gnorm = _grad_norm(autoencoder.decoder)

                writer.add_scalar("Loss/val_quick", val_loss, n_iter)
                writer.add_scalar("SNR/val_quick", val_snr, n_iter)
                writer.add_scalar("GradNorm/encoder", enc_gnorm, n_iter)
                writer.add_scalar("GradNorm/decoder", dec_gnorm, n_iter)
                if recent_loss > 0:
                    writer.add_scalar(
                        "Ratio/val_to_train", val_loss / recent_loss, n_iter
                    )

        return recent_loss, n_iter

    # Claude vibed me some startup banner code, so here it is:
    # ── Startup banner ────────────────────────────────────────────────────────────
    def _print_startup_banner() -> None:
        """Print a formatted training-configuration summary to stdout."""
        W = 64
        COL = 18  # label column width

        def hr(ch: str = "─") -> str:
            return "  " + ch * W

        def row(label: str, value: str) -> str:
            return f"  {label:<{COL}}{value}"

        # ── noise-type summary ────────────────────────────────────────────────────
        nt = hp.noise_types
        if nt is DEMANDNoiseType.ALL:
            env_names = [
                n.replace("_16k", "") for n in DEMANDNoiseType.ALL.value
            ]
            noise_header = f"ALL  ({len(env_names)} environments)"
        else:
            env_names = [t.name for t in nt]
            noise_header = f"{len(env_names)} environment(s)"
        env_lines = textwrap.wrap("  ".join(env_names), width=W - COL - 2)

        total_params = sum(p.numel() for p in autoencoder.parameters())
        lr = optimizer.defaults["lr"]
        wd = optimizer.defaults["weight_decay"]
        started_at = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(timestamp)
        )

        print()
        print(f"  ╔{'═' * W}╗")
        print(f"  ║{'  Denoising AutoEncoder – Training'.center(W)}║")
        print(f"  ║{started_at.center(W)}║")
        print(f"  ╚{'═' * W}╝")

        print()
        print("  DATA")
        print(hr())
        print(row("Train path", str(hp.train_data_path)))
        print(
            row(
                "Train files",
                f"{len(librispeech_train._source_flac_paths):,} FLAC",
            )
        )
        print(row("Test path", str(hp.test_data_path)))
        print(
            row(
                "Test files",
                f"{len(librispeech_test._source_flac_paths):,} FLAC",
            )
        )

        print()
        print("  FEATURES")
        print(hr())
        print(row("window_length", f"{hp.window_length} samples"))
        print(row("hop_length", f"{hp.hop_length} samples"))
        print(row("sample shape", str(librispeech_train.sample_shape)))

        print()
        print("  NOISE AUGMENTATION")
        print(hr())
        for i, line in enumerate(env_lines):
            print(row(noise_header if i == 0 else "", line))

        print()
        print("  MODEL")
        print(hr())
        print(row("Architecture", type(autoencoder).__name__))
        print(row("Input dim", f"{librispeech_train.sample_shape[0]:,}"))
        print(row("Latent dim", "128"))
        print(row("Total params", f"{total_params:,}"))

        print()
        print("  TRAINING")
        print(hr())
        print(row("Epochs", str(hp.n_epochs)))
        print(row("Batch size", str(hp.batch_size)))
        print(row("Optimizer", f"Adam  lr={lr:.0e}  weight_decay={wd:.0e}"))
        print(
            row(
                "LR scheduler",
                f"ReduceLROnPlateau  patience={hp.lr_patience}  "
                f"factor={hp.lr_factor}  min_lr={hp.lr_min:.0e}",
            )
        )
        print(row("Loss", type(loss_fn).__name__))
        print(
            row(
                "Val every",
                f"{hp.val_every} batches  ({hp.val_batches} per quick pass)",
            )
        )
        print(row("TensorBoard", writer.log_dir))

        print()
        print("  " + "═" * W)
        print("  Starting training …")
        print("  " + "═" * W)
        print()

    # Training loop with periodic validation and TensorBoard logging.
    _print_startup_banner()

    for epoch in range(hp.n_epochs):
        autoencoder.train()

        avg_loss, n_iter = train_epoch(n_iter, epoch, writer)

        # Full end-of-epoch validation
        epoch_val_loss, epoch_val_snr = run_quick_val(max_batches=10_000)

        print(
            "Epoch {} done | val loss: {:.6f} | SNR imp: {:.2f} dB".format(
                epoch, epoch_val_loss, epoch_val_snr
            )
        )

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/val_epoch", epoch_val_loss, epoch + 1)
        writer.add_scalar("SNR/val_epoch", epoch_val_snr, epoch + 1)
        writer.add_scalar("LR", current_lr, epoch + 1)
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": epoch_val_loss},
            epoch + 1,
        )
        writer.flush()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_path = f"{hp.model_dir}/{hp.name}.pth"
            torch.save(autoencoder.state_dict(), model_path)
