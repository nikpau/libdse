"""Training script for :class:`nets.WaveUNet` on the
LibriSpeech train-clean-100 corpus with DEMAND noise augmentation.

The model is trained to recover a clean raw waveform from its noisy
counterpart (end-to-end waveform denoising with Wave-U-Net).

Pipeline
--------
* Every batch: compute MSE loss, back-prop, log train loss + gradient norm.
* Every ``hp.val_every`` batches: run a quick validation pass over
  ``hp.val_batches`` batches of the test set and log val loss, SNR improvement,
  and the ratio of val loss to train loss.
* After each epoch: full validation, model checkpoint if val loss improves.

TensorBoard metrics
-------------------

- **Loss/train** — MSE on the current training mini-batch (smoothed over ``hp.log_every`` batches).
- **Loss/val_quick** — MSE on ``hp.val_batches`` val batches (logged every ``hp.val_every`` train batches).
- **SNR/val_quick** — Signal-to-noise ratio improvement (dB) on the quick val pass: 10·log₁₀(input_noise_power / residual_power).
- **Ratio/val_to_train** — val_loss / recent_train_loss; tracks over-fitting.
- **GradNorm/encoder** — L2 norm of encoder downsampling-path gradients (health check).
- **GradNorm/decoder** — L2 norm of decoder upsampling-path gradients (health check).
- **Loss/val_epoch** — Full val-set MSE at end of each epoch.
- **SNR/val_epoch** — Full val-set SNR improvement at end of each epoch.
"""

import torch
import libdse.nets as nets
import libdse.data.librispeech as data
from libdse.data.features import RawWaveformExtractor
from libdse.data.noise import DEMANDNoiseDataset, DEMANDNoiseType
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
    name: str = "waveunet"
    info: str = (
        "Wave-U-Net for speech enhancement, following the the architecture from"
        "Craig & Weyde (2018) 'Improved Speech Enhancement with the Wave-U-Net'.",
    )
    model_dir: str = "models"

    # Training Hyperparameters
    n_epochs: int = 40
    batch_size: int = 16
    val_every: int = (
        100  # run a quick val pass every this many training batches
    )
    val_batches: int = 50  # number of val batches used for the quick pass
    log_every: int = 100  # smooth train loss over this many batches

    # Data Hyperparameters
    train_data_path: Path = Path("data/train-clean-100")
    test_data_path: Path = Path("data/test-clean")
    noise_types: list[DEMANDNoiseType] = DEMANDNoiseType.ALL
    DEMAND_entry_point: Path = Path("data/noise/DEMAND")
    sampling_rate: int = 16_000  # Paper resamples to 8 kHz.

    # LR Scheduler Hyperparameters (ReduceLROnPlateau)
    lr_patience: int = 5  # epochs with no improvement before reducing LR
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
        window_length=16_384,  # nearest power of 2 to 1 second at 16 kHz
    )
    train_extractor = RawWaveformExtractor(**_extractor_kwargs, noise=noise_ds)
    test_extractor = RawWaveformExtractor(**_extractor_kwargs, noise=noise_ds)

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

    waveunet = nets.WaveUNet(
        n_layers=7,  # 10 layers with f_d=15 decimates a 16 384-sample window
        # down to only 2 samples at the bottleneck — below the
        # kernel size.  7 layers leaves ~114 samples there.
        f_d=15,  # Kernel size (number of filters) in the downsampling path.
        f_u=5,  # Kernel size (number of filters) in the upsampling path.
        F_c=16,  # Base number of filters in the first layer; this is multiplied by 2^i in the i-th layer (i=0 is the first layer).
    )
    optimizer = Adam(waveunet.parameters(), lr=1e-4, weight_decay=1e-5)
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
        waveunet.eval()
        total_loss = 0.0
        total_snr = 0.0
        n_batches = 0

        with torch.no_grad():
            for vinputs, vlabels in test_data:
                if n_batches >= max_batches:
                    break
                vinputs = vinputs.float().unsqueeze(1)  # (B, T) → (B, 1, T)
                vlabels = vlabels.float()  # (B, T)
                vforeground, _ = waveunet(vinputs)  # (B, 1, T')
                vforeground = vforeground.squeeze(1)  # (B, T')
                T_out = vforeground.shape[-1]
                start = (vlabels.shape[-1] - T_out) // 2
                vlabels = vlabels[:, start : start + T_out]  # (B, T')
                vinputs_crop = vinputs.squeeze(1)[:, start : start + T_out]
                total_loss += loss_fn(vforeground, vlabels).item()
                total_snr += _snr_improvement_db(
                    vinputs_crop, vforeground, vlabels
                )
                n_batches += 1

        waveunet.train()
        avg_loss = total_loss / max(n_batches, 1)
        avg_snr = total_snr / max(n_batches, 1)
        return avg_loss, avg_snr

    def train_epoch(
        n_iter: int, epoch_number: int, writer: SummaryWriter
    ) -> tuple[float, int]:
        """Train for one full epoch and return ``(avg_train_loss, n_iter)``.

        Every ``hp.log_every`` batches the smoothed training loss is written to
        TensorBoard.  Every ``hp.val_every`` batches a quick partial validation is
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

            input = input.float().unsqueeze(1)  # (B, T) → (B, 1, T)
            label = label.float()  # (B, T)
            foreground, _ = waveunet(input)  # (B, 1, T')
            foreground = foreground.squeeze(1)  # (B, T')
            T_out = foreground.shape[-1]
            start = (label.shape[-1] - T_out) // 2
            label = label[:, start : start + T_out]  # (B, T')
            loss: torch.Tensor = loss_fn(foreground, label)
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
                enc_gnorm = _grad_norm(waveunet.encoder)
                dec_gnorm = _grad_norm(waveunet.decoder)

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

        total_params = sum(p.numel() for p in waveunet.parameters())
        lr = optimizer.defaults["lr"]
        wd = optimizer.defaults["weight_decay"]
        started_at = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(timestamp)
        )

        print()
        print(f"  ╔{'═' * W}╗")
        print(f"  ║{'  Wave-U-Net – Speech Enhancement Training'.center(W)}║")
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
        print(row("Domain", "raw waveform (no STFT)"))
        print(
            row(
                "window_length",
                f"{_extractor_kwargs['window_length']:,} samples",
            )
        )
        print(row("sample shape", str(librispeech_train.sample_shape)))

        print()
        print("  NOISE AUGMENTATION")
        print(hr())
        for i, line in enumerate(env_lines):
            print(row(noise_header if i == 0 else "", line))

        print()
        print("  MODEL")
        print(hr())
        print(row("Architecture", type(waveunet).__name__))
        print(
            row("Input dim", f"{librispeech_train.sample_shape[0]:,} samples")
        )
        print(row("n_layers", str(waveunet.n_layers)))
        print(row("F_c (base ch.)", str(waveunet.F_c)))
        print(row("f_d (enc kernel)", str(waveunet.f_d)))
        print(row("f_u (dec kernel)", str(waveunet.f_u)))
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
        waveunet.train()

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
            torch.save(waveunet.state_dict(), model_path)
