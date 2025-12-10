#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""Interactive generator of custom Threshold-of-Hearing (ToH) tables.

The tool plays dyadic subband noise bursts and asks the listener whether the
distortion is audible. The largest inaudible quantization step per subband is
stored in ``custom_ToH.txt`` (or the file passed through ``--toh-output``).
These measurements are intended for future integrations such as
``dyadic_linear_ToH.py``.

Answer the prompt with:

* ``y`` when the noise is audible (the amplitude will decrease)
* ``n`` when it is not audible (the amplitude will increase)
* ``a`` to accept the current best inaudible value
* ``b`` to type an explicit amplitude
* ``r`` to replay the last trial
* ``s`` to keep the stored value for that subband
* ``q`` to abort the session

Combine ``--toh-subband`` with ``--toh-load`` to refine specific bands without
redoing the entire measurement.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pywt
import sounddevice as sd

import minimal


def _ensure_minimal_quant_argument() -> None:
    for action in minimal.parser._actions:
        if "--minimal_quantization_step_size" in action.option_strings:
            return
    minimal.parser.add_argument(
        "-q",
        "--minimal_quantization_step_size",
        type=int,
        default=128,
        help="Minimal quantization step size used as the floor for ToH amplitudes.",
    )


_ensure_minimal_quant_argument()


def _ensure_wavelet_arguments() -> None:
    existing = {opt for action in minimal.parser._actions for opt in action.option_strings}
    if "--wavelet_name" not in existing:
        minimal.parser.add_argument(
            "-w",
            "--wavelet_name",
            type=str,
            default="db5",
            help="Wavelet family name used for the dyadic decomposition (default: db5).",
        )
    if "--levels" not in existing:
        minimal.parser.add_argument(
            "-e",
            "--levels",
            type=int,
            default=6,
            help="Number of DWT levels to analyse (default: 6).",
        )


_ensure_wavelet_arguments()

minimal.parser.add_argument(
    "--toh-output",
    type=str,
    default="custom_ToH.txt",
    help="Destination file for the generated ToH table.",
)
minimal.parser.add_argument(
    "--toh-load",
    type=str,
    help="Optional existing ToH table used as seeds for the calibration.",
)
minimal.parser.add_argument(
    "--toh-subband",
    type=str,
    help=(
        "Comma-separated list of subband indices to calibrate. When omitted, "
        "all subbands are processed in ascending frequency order."
    ),
)
minimal.parser.add_argument(
    "--toh-noise-seconds",
    type=float,
    default=1.0,
    help="Duration (seconds) of the noise segment reproduced on each trial.",
)
minimal.parser.add_argument(
    "--toh-silence-seconds",
    type=float,
    default=1.0,
    help="Duration (seconds) of the silence segment following the noise.",
)
minimal.parser.add_argument(
    "--toh-pattern-repeats",
    type=int,
    default=2,
    help="Number of noise/silence alternations played per trial.",
)
minimal.parser.add_argument(
    "--toh-increase-factor",
    type=float,
    default=1.2,
    help="Multiplicative factor applied when the noise is inaudible.",
)
minimal.parser.add_argument(
    "--toh-decrease-factor",
    type=float,
    default=0.6,
    help="Multiplicative factor applied when the noise is audible.",
)
minimal.parser.add_argument(
    "--toh-initial-step",
    type=float,
    default=0.0,
    help="Initial quantization step guess (0 delegates to the minimal step).",
)
minimal.parser.add_argument(
    "--toh-max-step",
    type=float,
    default=2_000_000.0,
    help="Upper bound for the quantization step during calibration.",
)
minimal.parser.add_argument(
    "--toh-min-step",
    type=float,
    default=0.0,
    help="Lower bound for the quantization step during calibration (default: 0).",
)


@dataclass
class SubbandInfo:
    index: int
    label: str
    freq_start: float
    freq_end: float
    slc: slice
    length: int

    @property
    def bandwidth(self) -> float:
        return self.freq_end - self.freq_start


class CustomToHBuilder:
    """Encapsulates the interactive calibration routine."""

    def __init__(self) -> None:
        self.frames_per_chunk = int(minimal.args.frames_per_chunk)
        self.num_channels = int(minimal.args.number_of_channels)
        self.samplerate = float(minimal.args.frames_per_second)
        self.noise_seconds = max(0.1, float(minimal.args.toh_noise_seconds))
        self.silence_seconds = max(0.1, float(minimal.args.toh_silence_seconds))
        self.pattern_repeats = max(1, int(minimal.args.toh_pattern_repeats))
        self.increase_factor = max(1.01, float(minimal.args.toh_increase_factor))
        self.decrease_factor = min(0.99, float(minimal.args.toh_decrease_factor))
        self.min_step = max(0.0, float(minimal.args.toh_min_step))
        self.max_step = max(self.min_step + 1.0, float(minimal.args.toh_max_step))
        self.initial_step = float(minimal.args.toh_initial_step)
        self.minimal_step = float(max(1, minimal.args.minimal_quantization_step_size))
        self.tolerance = max(1.0, self.minimal_step * 0.05)
        self.output_device = minimal.args.output_device

        if self.decrease_factor <= 0.0:
            raise ValueError("--toh-decrease-factor must be positive and < 1")
        if self.increase_factor <= 1.0:
            raise ValueError("--toh-increase-factor must be greater than 1")

        self._rng = np.random.default_rng()

        self.wavelet = pywt.Wavelet(minimal.args.wavelet_name)
        self._init_wavelet_metadata()
        self.subbands = self._build_subbands()
        self.templates = self._generate_templates()

        self.base_coeffs = np.zeros((self.coeff_length, self.num_channels), dtype=np.float64)
        self.results: List[Optional[float]] = [None] * len(self.subbands)

        if minimal.args.toh_load:
            self._load_existing(Path(minimal.args.toh_load))

        if self.output_device is not None:
            sd.default.device = self.output_device
        sd.default.channels = self.num_channels
        sd.default.samplerate = self.samplerate
        sd.default.dtype = "int16"
        sd.default.blocksize = self.frames_per_chunk

        self.int16_min = np.iinfo(np.int16).min
        self.int16_max = np.iinfo(np.int16).max

    def _init_wavelet_metadata(self) -> None:
        max_filter = max(self.wavelet.dec_len, self.wavelet.rec_len)
        default_levels = pywt.dwt_max_level(
            data_len=max(1, self.frames_per_chunk // 4),
            filter_len=max_filter,
        )
        requested_levels = int(minimal.args.levels) if minimal.args.levels else default_levels
        self.dwt_levels = max(1, requested_levels)

        zero_chunk = np.zeros(self.frames_per_chunk, dtype=np.float64)
        coeffs = pywt.wavedec(zero_chunk, wavelet=self.wavelet, level=self.dwt_levels, mode="per")
        coeff_array, self.slices = pywt.coeffs_to_array(coeffs)
        self.coeff_length = int(coeff_array.size)
        if self.coeff_length != self.frames_per_chunk:
            logging.warning(
                "Coefficient vector length (%d) differs from frames_per_chunk (%d).",
                self.coeff_length,
                self.frames_per_chunk,
            )

    def _build_subbands(self) -> List[SubbandInfo]:
        nyquist = self.samplerate / 2.0
        subbands: List[SubbandInfo] = []
        coeff_index = np.arange(self.coeff_length, dtype=np.int32)

        approx_slice = self.slices[0][0]
        approx_len = int(coeff_index[approx_slice].size)
        approx_end = nyquist / (2 ** self.dwt_levels)
        subbands.append(
            SubbandInfo(
                index=0,
                label="A",
                freq_start=0.0,
                freq_end=approx_end,
                slc=approx_slice,
                length=approx_len,
            )
        )

        for level_idx in range(self.dwt_levels):
            detail_slice = self.slices[level_idx + 1]["d"][0]
            detail_level = self.dwt_levels - level_idx
            freq_start = nyquist / (2 ** detail_level)
            freq_end = nyquist if detail_level == 1 else nyquist / (2 ** (detail_level - 1))
            detail_len = int(coeff_index[detail_slice].size)
            subbands.append(
                SubbandInfo(
                    index=len(subbands),
                    label=f"D{detail_level}",
                    freq_start=freq_start,
                    freq_end=freq_end,
                    slc=detail_slice,
                    length=detail_len,
                )
            )

        return subbands

    def _generate_templates(self) -> List[np.ndarray]:
        templates: List[np.ndarray] = []
        for info in self.subbands:
            # Fixed random template keeps the noise shape stable across trials.
            template = self._rng.uniform(-0.5, 0.5, size=(info.length, self.num_channels))
            templates.append(template.astype(np.float64, copy=False))
        return templates

    def _load_existing(self, path: Path) -> None:
        if not path.exists():
            logging.warning("Existing ToH file '%s' not found.", path)
            return

        values: List[float] = []
        try:
            with path.open("r", encoding="ascii", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    try:
                        qss_value = float(parts[2])
                    except ValueError:
                        continue
                    values.append(qss_value)
        except OSError as exc:
            logging.warning("Unable to read '%s': %s", path, exc)
            return

        if not values:
            logging.warning("Existing ToH file '%s' looks empty.", path)
            return

        if len(values) != len(self.subbands):
            logging.warning(
                "Loaded %d ToH values, but %d subbands are expected. Truncating the list.",
                len(values),
                len(self.subbands),
            )

        for idx, info in enumerate(self.subbands):
            if idx >= len(values):
                break
            qss = float(values[idx])
            self.results[idx] = qss
            self.base_coeffs[info.slc, :] = self.templates[idx] * qss
        logging.info(
            "Loaded ToH data for %d subbands from '%s'.",
            min(len(values), len(self.subbands)),
            path,
        )

    def run(self) -> None:
        targets = self._resolve_targets()
        if not targets:
            logging.info("No subbands selected for calibration.")
            return

        logging.info(
            "Processing %d subband(s): %s",
            len(targets),
            ", ".join(str(idx) for idx in targets),
        )

        try:
            for idx in targets:
                self.calibrate_subband(idx)
        finally:
            try:
                sd.stop()
            except Exception:
                pass

        self.save_results()

    def _resolve_targets(self) -> List[int]:
        if not minimal.args.toh_subband:
            return list(range(len(self.subbands)))
        raw_items = [item.strip() for item in minimal.args.toh_subband.split(",") if item.strip()]
        targets: List[int] = []
        for item in raw_items:
            try:
                value = int(item)
            except ValueError:
                logging.warning("Ignoring invalid subband index '%s'.", item)
                continue
            if value < 0 or value >= len(self.subbands):
                logging.warning("Subband index %d is out of range (0-%d).", value, len(self.subbands) - 1)
                continue
            if value not in targets:
                targets.append(value)
        targets.sort()
        return targets

    def calibrate_subband(self, idx: int) -> None:
        info = self.subbands[idx]
        current = self.results[idx] if self.results[idx] is not None else (
            self.initial_step if self.initial_step > 0.0 else self.minimal_step
        )
        current = float(np.clip(current, self.min_step or 0.0, self.max_step))
        lower: Optional[float] = self.results[idx]
        upper: Optional[float] = None
        accepted: Optional[float] = None
        manual_only = False
        need_play = True

        if idx > 0:
            missing_prev = [j for j in range(idx) if self.results[j] is None]
            if missing_prev:
                logging.warning(
                    "Previous subbands without stored ToH values: %s (their noise is assumed zero)",
                    ", ".join(str(i) for i in missing_prev),
                )

        logging.info(
            "Subband %d (%s): %.2f Hz - %.2f Hz (bandwidth %.2f Hz, %d coefficients)",
            info.index,
            info.label,
            info.freq_start,
            info.freq_end,
            info.bandwidth,
            info.length,
        )

        while True:
            logging.info(
                "Trial amplitude %.4f | lower=%s upper=%s",
                current,
                f"{lower:.4f}" if lower is not None else "-",
                f"{upper:.4f}" if upper is not None else "-",
            )

            if not manual_only and need_play:
                try:
                    self._play_trial(idx, current)
                except Exception as exc:
                    logging.error("Audio playback failed: %s", exc)
                    raise
                need_play = False

            response = input(
                "Audible? [y] yes / [n] no / [a] accept / [b] set / [r] repeat / [s] skip / [q] quit > "
            ).strip().lower()

            if manual_only and response in {"y", "yes", "n", "no"}:
                logging.info(
                    "Manual resolution required: use 'b' to enter a value, 'a' to accept, 's' to keep stored, or 'q' to abort."
                )
                continue

            if response in {"y", "yes"}:
                upper = current
                if lower is not None and math.isclose(lower, current, rel_tol=1e-9, abs_tol=1e-9):
                    lower = None

                if lower is not None:
                    if upper is None or upper <= lower:
                        candidate = lower * self.decrease_factor
                    else:
                        candidate = (lower + upper) / 2.0
                    if abs(candidate - lower) <= self.tolerance and upper is not None and upper > lower:
                        accepted = lower
                        logging.info(
                            "Reached tolerance %.2f. Accepting %.4f.",
                            self.tolerance,
                            accepted,
                        )
                        break
                if lower is None:
                    candidate = current * self.decrease_factor

                new_current = max(self.min_step, candidate)
                if math.isclose(new_current, current, rel_tol=1e-12, abs_tol=1e-12):
                    logging.info(
                        "Amplitude already at the configured minimum (%.4f). Use 'b' to set a custom value, 's' to keep it, 'a' to accept, or 'q' to abort.",
                        current,
                    )
                    manual_only = True
                    need_play = False
                    continue

                current = new_current
                manual_only = False
                need_play = True
            elif response in {"n", "no"}:
                lower = current
                current = min(self.max_step, current * self.increase_factor)
                manual_only = False
                need_play = True
            elif response in {"a", "accept"}:
                accepted = lower if lower is not None else current
                break
            elif response in {"b", "set"}:
                value_str = input("New amplitude value: ").strip()
                try:
                    candidate = float(value_str)
                except ValueError:
                    logging.warning("Invalid amplitude '%s'.", value_str)
                    continue
                current = float(np.clip(candidate, self.min_step, self.max_step))
                manual_only = False
                need_play = True
            elif response in {"r", "repeat"}:
                manual_only = False
                need_play = True
                continue
            elif response in {"s", "skip"}:
                if self.results[idx] is not None:
                    accepted = self.results[idx]
                break
            elif response in {"q", "quit"}:
                raise KeyboardInterrupt
            else:
                logging.info("Please answer with y, n, a, b, r, s, or q.")
                continue

        if accepted is None:
            logging.info("No update stored for subband %d.", idx)
            return

        self.results[idx] = accepted
        self.base_coeffs[info.slc, :] = self.templates[idx] * accepted
        logging.info("Stored ToH amplitude %.4f for subband %d.", accepted, idx)

    def _play_trial(self, subband_idx: int, amplitude: float) -> None:
        coeffs_view = self.base_coeffs.copy()
        coeffs_view[self.subbands[subband_idx].slc, :] = self.templates[subband_idx] * amplitude
        chunk = self._coeffs_to_time(coeffs_view)

        noise_segment = self._repeat_to_length(chunk, self.noise_seconds)
        silence_frames = max(1, int(round(self.silence_seconds * self.samplerate)))
        silence_segment = np.zeros((silence_frames, self.num_channels), dtype=np.float64)

        pattern = []
        for _ in range(self.pattern_repeats):
            pattern.append(noise_segment)
            pattern.append(silence_segment)
        playback = np.vstack(pattern)

        audio = np.clip(np.rint(playback), self.int16_min, self.int16_max).astype(np.int16, copy=False)
        try:
            sd.stop()
        except Exception:
            pass
        sd.play(audio, samplerate=int(self.samplerate))
        sd.wait()

    def _repeat_to_length(self, chunk: np.ndarray, seconds: float) -> np.ndarray:
        target_frames = max(1, int(round(seconds * self.samplerate)))
        repeats = max(1, int(math.ceil(target_frames / chunk.shape[0])))
        tiled = np.tile(chunk, (repeats, 1))
        return tiled[:target_frames]

    def _coeffs_to_time(self, coeff_array: np.ndarray) -> np.ndarray:
        chunk = np.empty((self.frames_per_chunk, self.num_channels), dtype=np.float64)
        for channel in range(self.num_channels):
            coeffs = pywt.array_to_coeffs(
                coeff_array[:, channel],
                self.slices,
                output_format="wavedec",
            )
            chunk[:, channel] = pywt.waverec(coeffs, wavelet=self.wavelet, mode="per")
        return chunk

    def save_results(self) -> None:
        missing = [info.index for info, value in zip(self.subbands, self.results) if value is None]
        if missing:
            raise RuntimeError(
                "Cannot emit ToH table; missing values for subbands: "
                + ", ".join(str(idx) for idx in missing)
            )

        output_path = Path(minimal.args.toh_output).expanduser()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        with output_path.open("w", encoding="ascii") as handle:
            handle.write("# Initial_frequency_in_Hertz Band-width_in_Hertz ToH\n")
            for info, value in zip(self.subbands, self.results):
                handle.write(
                    f"{info.freq_start:>12.2f}\t{info.bandwidth:>12.2f}\t{value:.4f}\n"
                )
        logging.info("Custom ToH profile stored in '%s'.", output_path)


try:  # pragma: no cover
    import argcomplete  # type: ignore
except ImportError:  # pragma: no cover
    argcomplete = None  # type: ignore
    logging.warning("argcomplete not available.")


if __name__ == "__main__":
    minimal.parser.description = __doc__
    if argcomplete:  # pragma: no cover
        try:
            argcomplete.autocomplete(minimal.parser)  # type: ignore[attr-defined]
        except Exception:
            logging.warning("argcomplete autocomplete failed.")

    minimal.args = minimal.parser.parse_known_args()[0]

    try:
        builder = CustomToHBuilder()
        builder.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nCalibration interrupted by user.")
    except RuntimeError as exc:
        logging.error("%s", exc)
        minimal.parser.exit(1)
