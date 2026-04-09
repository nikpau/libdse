from enum import Enum
from pathlib import Path
import librosa
import numpy as np
from numpy.typing import NDArray

from dae.data.err import EntryPointError


class DEMANDNoiseType(Enum):
    """Folder-name identifiers for the `DEMAND <https://doi.org/10.5281/zenodo.1227120>`_ noise dataset.

    Each member's value is the exact directory name inside the DEMAND
    archive, which follows the pattern ``<CATEGORY><NAME>_<FS>k``.  Pass a
    sub-set of members to :class:`DEMANDNoiseDataset` to restrict which
    noise environments are loaded.
    """

    KITCHEN = "DKITCHEN_16k"
    LIVING = "DLIVING_16k"
    WASHING = "DWASHING_16k"
    FIELD = "NFIELD_16k"
    PARK = "NPARK_16k"
    RIVER = "NRIVER_16k"
    HALLWAY = "OHALLWAY_16k"
    MEETING = "OMEETING_16k"
    OFFICE = "OOFFICE_16k"
    CAFETERIA = "PCAFETER_16k"
    RESTAURANT = "PRESTO_16k"
    STATION = "PSTATION_16k"
    SQUARE = "SPSQUARE_16k"
    TRAFFIC = "STRAFFIC_16k"
    BUS = "TBUS_16k"
    CAR = "TCAR_16k"
    METRO = "TMETRO_16k"


class DEMANDNoiseDataset:
    """Loads and exposes DEMAND background-noise recordings as a single array.

    The `DEMAND <https://doi.org/10.5281/zenodo.1227120>`_ dataset contains
    18 real-world noise environments, each recorded on 16 channels at 16 kHz.
    Only channel 1 (``ch1.wav``) is used here.  All selected recordings are
    concatenated end-to-end into :attr:`noise` so that callers can slice
    arbitrary-length segments without managing individual files.

    :param entry_point: Directory that directly contains the per-environment
        sub-directories (e.g. ``DKITCHEN_16k/``, ``TCAR_16k/``, …).
    :type entry_point: :class:`pathlib.Path`
    :param noise_types: Noise environments to load.  Every requested type
        must have a matching sub-directory under *entry_point*.
    :type noise_types: list[:class:`DEMANDNoiseType`]

    :raises EntryPointError: If any requested environment directory is missing
        under *entry_point*.
    """

    def __init__(
        self, entry_point: Path, noise_types: list[DEMANDNoiseType]
    ) -> None:
        """Validate *entry_point* and load all requested noise recordings.

        :param entry_point: Root directory of the DEMAND dataset.
        :type entry_point: :class:`pathlib.Path`
        :param noise_types: Environments to include.
        :type noise_types: list[:class:`DEMANDNoiseType`]
        :raises EntryPointError: If a required environment directory is absent.
        """
        # Each environment is a 16-channel recording; only channel 1 is used.
        filename = "ch01.wav"

        self.fs = 16_000  # 16 kHz — fixed for the whole DEMAND dataset

        all_dirs = [d.name for d in entry_point.iterdir()]
        required_noise_type_dirs = [t.value for t in noise_types]
        if not all(d in all_dirs for d in required_noise_type_dirs):
            raise EntryPointError(
                "DEMAND entry point is missing required environment "
                f"directories.\n\n"
                f"Available: {sorted(all_dirs)}\n"
                f"Requested: {sorted(required_noise_type_dirs)}"
            )

        data_dirs = [
            directory
            for directory in entry_point.iterdir()
            if directory.name in required_noise_type_dirs
        ]

        self.target_files = []
        for directory in data_dirs:
            self.target_files.extend(directory.rglob(filename))

        self.noise = self._expand_noise()

    def __repr__(self) -> str:
        """Return a concise string representation.

        :return: ``DEMANDNoiseDataset(fs=F, noise_samples=N)``
        :rtype: str
        """
        return (
            f"DEMANDNoiseDataset(fs={self.fs}, noise_samples={len(self.noise)})"
        )

    def _expand_noise(self) -> NDArray[np.float32]:
        """Load all target WAV files and concatenate them into a single array.

        :return: 1-D float32 array of all noise samples concatenated in order.
        :rtype: :class:`numpy.ndarray`
        """
        samples = []
        for f in self.target_files:
            y, _ = librosa.load(f, sr=self.fs, mono=True)
            samples.append(y)
        return np.ravel(samples)


def add_noise_snr(
    signal: NDArray[np.float32], noise: NDArray[np.float32], snr_db: float
) -> NDArray[np.float32]:
    """Mix *noise* into *signal* at a target signal-to-noise ratio.

    The noise array is first padded (wrap mode) or truncated to match the
    length of *signal*, then scaled so that the resulting SNR equals
    *snr_db*.  If the mixture clips (peak > 1.0) it is peak-normalised.

    .. math::

        \\text{SNR}_{\\text{dB}} = 10 \\log_{10}\\!
            \\left(\\frac{P_{\\text{signal}}}{P_{\\text{noise}}}\\right)

    :param signal: Clean mono waveform, assumed to be in ``[-1, 1]``.
    :type signal: :class:`numpy.ndarray` of float32
    :param noise: Noise waveform.  May be shorter or longer than *signal*.
    :type noise: :class:`numpy.ndarray` of float32
    :param snr_db: Desired signal-to-noise ratio in decibels.
    :type snr_db: float
    :return: Noisy mixture with the same length as *signal*, peak-normalised
        if clipping occurs.
    :rtype: :class:`numpy.ndarray` of float32
    """

    # Match noise length to signal (wrap-pad if shorter, truncate if longer).
    if len(noise) < len(signal):
        noise = np.pad(noise, (0, len(signal) - len(noise)), "wrap")
    else:
        noise = noise[: len(signal)]

    # Power = mean squared amplitude.
    p_signal = np.mean(signal**2)
    p_noise = np.mean(noise**2)

    # Derive the noise power that satisfies the target SNR, then scale.
    p_target_noise = p_signal / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(p_target_noise / p_noise)

    noisy_signal = signal + (noise * scaling_factor)

    # Peak-normalise to prevent clipping.
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 1.0:
        noisy_signal = noisy_signal / max_val

    return noisy_signal
