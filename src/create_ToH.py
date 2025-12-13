#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""Interactive generator of custom Threshold-of-Hearing (ToH) tables.

Steps (per the *Custom ToH* milestone):

1. Inject uniform quantisation noise into one dyadic subband at a time.
2. Ask the listener whether the noise is audible while adapting its amplitude.
3. Keep the highest *inaudible* levels found for previous subbands as we move
   towards higher frequencies.

The resulting table (``custom_ToH.txt`` by default) can be shared with the
interlocutor so that codecs such as ``dyadic_linear_ToH`` adapt their
quantisation steps to the real sensitivity of the listener.
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


@dataclass
class Subband:
    index: int
    label: str
    freq_start: float
    freq_end: float
    slc: slice
    length: int

    @property
    def bandwidth(self) -> float:
        return self.freq_end - self.freq_start


def _ensure_wavelet_arguments() -> None:
    existing = {opt for action in minimal.parser._actions for opt in action.option_strings}
    if "--wavelet_name" not in existing:
        minimal.parser.add_argument(
            "-w",
            "--wavelet_name",
            type=str,
            default="db5",
            help="Wavelet used for the dyadic analysis (default: db5).",
        )
    if "--levels" not in existing:
        minimal.parser.add_argument(
            "-e",
            "--levels",
            type=int,
            default=6,
            help="Number of DWT levels processed during calibration (default: 6).",
        )


def _ensure_quant_step_argument() -> None:
    for action in minimal.parser._actions:
        if "--minimal_quantization_step_size" in action.option_strings:
            return
    minimal.parser.add_argument(
        "-q",
        "--minimal_quantization_step_size",
        type=int,
        default=128,
        help="Minimal quantisation step (used as amplitude floor).",
    )


_ensure_wavelet_arguments()
_ensure_quant_step_argument()

minimal.parser.add_argument(
    "--toh-output",
    type=str,
    default="custom_ToH.txt",
    help="Destination file for the generated ToH table.",
)
minimal.parser.add_argument(
    "--toh-load",
    type=str,
    help="Optional existing ToH table used as seeds (skips calibrated subbands).",
)
minimal.parser.add_argument(
    "--toh-subband",
    type=str,
    help="Comma-separated list of subband indices to calibrate (default: all).",
)
minimal.parser.add_argument(
    "--toh-noise-seconds",
    type=float,
    default=1.0,
    help="Duration in seconds of the noise burst per trial (default: 1).",
)
minimal.parser.add_argument(
    "--toh-silence-seconds",
    type=float,
    default=1.0,
    help="Duration of the silence that follows the noise burst (default: 1).",
)
minimal.parser.add_argument(
    "--toh-pattern-repeats",
    type=int,
    default=2,
    help="Number of noise/silence alternations per trial (default: 2).",
)
minimal.parser.add_argument(
    "--toh-increase-factor",
    type=float,
    default=1.4,
    help="Factor applied when the noise remains inaudible (default: 1.4).",
)
minimal.parser.add_argument(
    "--toh-decrease-factor",
    type=float,
    default=0.5,
    help="Factor applied when the noise becomes audible (default: 0.5).",
)
minimal.parser.add_argument(
    "--toh-initial-step",
    type=float,
    default=0.0,
    help="Initial amplitude guess; 0 delegates to the minimal step.",
)
minimal.parser.add_argument(
    "--toh-max-step",
    type=float,
    default=1_000_000.0,
    help="Upper bound for the amplitude during calibration.",
)
minimal.parser.add_argument(
    "--toh-min-step",
    type=float,
    default=0.0,
    help="Lower bound for the amplitude during calibration (default: 0).",
)
minimal.parser.add_argument(
    "--toh-tolerance",
    type=float,
    default=0.0,
    help="Stop when upper-lower <= tolerance (0 -> 5% of the minimal step).",
)
minimal.parser.add_argument(
    "--toh-reversals",
    type=int,
    default=3,
    help="Number of direction reversals required before auto-accepting (default: 3).",
)
minimal.parser.add_argument(
    "--toh-rng-seed",
    type=int,
    default=5489,
    help="Seed for the random noise generator (fixed by default for repeatability).",
)


class CustomToHSession:
    """Encapsulates the interactive measurement workflow."""

    def __init__(self) -> None:
        self.frames_per_chunk = int(minimal.args.frames_per_chunk)
        self.channels = int(minimal.args.number_of_channels)
        self.sample_rate = float(minimal.args.frames_per_second)
        self.noise_seconds = max(0.1, float(minimal.args.toh_noise_seconds))
        self.silence_seconds = max(0.1, float(minimal.args.toh_silence_seconds))
        self.pattern_repeats = max(1, int(minimal.args.toh_pattern_repeats))
        self.increase_factor = max(1.01, float(minimal.args.toh_increase_factor))
        self.decrease_factor = min(0.99, float(minimal.args.toh_decrease_factor))
        self.min_step = max(0.0, float(minimal.args.toh_min_step))
        self.max_step = max(self.min_step + 1.0, float(minimal.args.toh_max_step))
        base_step = max(1, int(minimal.args.minimal_quantization_step_size))
        self.base_step = float(base_step)
        self.initial_step = float(minimal.args.toh_initial_step or self.base_step)
        self.tolerance = float(minimal.args.toh_tolerance or (0.05 * self.base_step))
        self.target_reversals = max(1, int(minimal.args.toh_reversals))
        self.output_device = minimal.args.output_device
        self.rng = np.random.default_rng(int(minimal.args.toh_rng_seed))

        if self.decrease_factor <= 0.0 or self.decrease_factor >= 1.0:
            raise ValueError("--toh-decrease-factor must lie in (0,1)")
        if self.increase_factor <= 1.0:
            raise ValueError("--toh-increase-factor must be greater than 1")

        self.wavelet = pywt.Wavelet(minimal.args.wavelet_name)
        self._init_wavelet_layout()
        self.subbands = self._build_subbands()
        self.templates = self._build_noise_templates()

        self.base_coeffs = np.zeros((self.coeff_length, self.channels), dtype=np.float64)
        self.results: List[Optional[float]] = [None] * len(self.subbands)

        if minimal.args.toh_load:
            self._load_existing(Path(minimal.args.toh_load))

        if self.output_device is not None:
            sd.default.device = self.output_device
        sd.default.channels = self.channels
        sd.default.samplerate = self.sample_rate
        sd.default.dtype = "int16"
        sd.default.blocksize = self.frames_per_chunk

        self.int16_min = np.iinfo(np.int16).min
        self.int16_max = np.iinfo(np.int16).max

    # ------------------------------------------------------------------
    # Wavelet helpers
    # ------------------------------------------------------------------
    def _init_wavelet_layout(self) -> None:
        data = np.zeros(self.frames_per_chunk, dtype=np.float64)
        max_levels = pywt.dwt_max_level(len(data), self.wavelet.dec_len)
        requested = max(1, int(minimal.args.levels))
        self.dwt_levels = min(requested, max_levels)
        coeffs = pywt.wavedec(data, wavelet=self.wavelet, level=self.dwt_levels, mode="per")
        coeff_array, self.slices = pywt.coeffs_to_array(coeffs)
        self.coeff_length = int(coeff_array.size)
        if self.coeff_length != self.frames_per_chunk:
            logging.warning(
                "Coefficient vector length (%d) differs from frames_per_chunk (%d).",
                self.coeff_length,
                self.frames_per_chunk,
            )

    def _build_subbands(self) -> List[Subband]:
        nyquist = self.sample_rate / 2.0
        coeff_index = np.arange(self.coeff_length, dtype=np.int32)
        subbands: List[Subband] = []

        approx_slice = self.slices[0][0]
        approx_len = int(coeff_index[approx_slice].size)
        approx_end = nyquist / (2 ** self.dwt_levels)
        subbands.append(
            Subband(
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
            start = nyquist / (2 ** detail_level)
            end = nyquist if detail_level == 1 else nyquist / (2 ** (detail_level - 1))
            detail_len = int(coeff_index[detail_slice].size)
            subbands.append(
                Subband(
                    index=len(subbands),
                    label=f"D{detail_level}",
                    freq_start=start,
                    freq_end=end,
                    slc=detail_slice,
                    length=detail_len,
                )
            )

        return subbands

    def _build_noise_templates(self) -> List[np.ndarray]:
        templates: List[np.ndarray] = []
        for info in self.subbands:
            template = self.rng.uniform(-0.5, 0.5, size=(info.length, self.channels))
            templates.append(template.astype(np.float64, copy=False))
        return templates

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
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
                        value = float(parts[2])
                    except ValueError:
                        continue
                    values.append(value)
        except OSError as exc:
            logging.warning("Unable to read '%s': %s", path, exc)
            return

        if not values:
            logging.warning("Existing ToH file '%s' looks empty.", path)
            return
        if len(values) != len(self.subbands):
            logging.warning(
                "Loaded %d ToH values, but %d subbands are expected. Truncating.",
                len(values),
                len(self.subbands),
            )

        for idx, value in enumerate(values[: len(self.subbands)]):
            self.results[idx] = value
            self.base_coeffs[self.subbands[idx].slc, :] = self.templates[idx] * value
        logging.info("Loaded seeds for %d subbands from '%s'.", min(len(values), len(self.subbands)), path)

    def save_results(self) -> None:
        missing = [info.index for info, value in zip(self.subbands, self.results) if value is None]
        if missing:
            raise RuntimeError(
                "Cannot store ToH table; missing values for subbands: "
                + ", ".join(str(idx) for idx in missing)
            )

        output_path = Path(minimal.args.toh_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="ascii") as handle:
            handle.write("# Initial_frequency_in_Hertz\tBand-width_in_Hertz\tToH\n")
            for info, value in zip(self.subbands, self.results):
                handle.write(
                    f"{info.freq_start:>12.2f}\t{info.bandwidth:>12.2f}\t{float(value):.4f}\n"
                )
        logging.info("Custom ToH profile stored in '%s'.", output_path)

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------
    def run(self) -> None:
        targets = self._resolve_targets()
        if not targets:
            logging.info("No subbands selected for calibration.")
            return

        logging.info(
            "Calibrating %d subband(s): %s",
            len(targets),
            ", ".join(str(idx) for idx in targets),
        )

        try:
            for idx in targets:
                self._calibrate_subband(idx)
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
        for token in raw_items:
            try:
                value = int(token)
            except ValueError:
                logging.warning("Ignoring invalid subband index '%s'.", token)
                continue
            if value < 0 or value >= len(self.subbands):
                logging.warning("Subband index %d out of range (0-%d).", value, len(self.subbands) - 1)
                continue
            if value not in targets:
                targets.append(value)
        targets.sort()
        return targets

    # ------------------------------------------------------------------
    # Calibration core
    # ------------------------------------------------------------------
    def _calibrate_subband(self, idx: int) -> None:
        info = self.subbands[idx]
        logging.info(
            "Subband %d (%s): %.2f Hz – %.2f Hz (bandwidth %.2f Hz, %d coeffs)",
            info.index,
            info.label,
            info.freq_start,
            info.freq_end,
            info.bandwidth,
            info.length,
        )

        lower = self.results[idx]
        upper: Optional[float] = None
        current = float(self.results[idx] or self.initial_step)
        current = float(np.clip(current, self.min_step or 0.0, self.max_step))
        if current <= 0.0:
            current = self.base_step

        reversals = 0
        last_direction: Optional[str] = None
        need_play = True
        trial_index = 1

        while True:
            if need_play:
                lower_txt = "-" if lower is None else f"{lower:.4f}"
                upper_txt = "-" if upper is None else f"{upper:.4f}"
                logging.info(
                    "Trial %d -> amp=%.4f | lower=%s | upper=%s | reversals=%d | stored=%s",
                    trial_index,
                    current,
                    lower_txt,
                    upper_txt,
                    reversals,
                    "-" if self.results[idx] is None else f"{self.results[idx]:.4f}",
                )
                self._play_trial(idx, current)
                need_play = False
                trial_index += 1

            response = input(
                "Audible? [y] yes / [n] no / [a] accept / [b] set / [r] repeat / [s] skip / [q] quit > "
            ).strip().lower()

            if response in {"y", "yes"}:
                upper = current
                direction = "down"
                if lower is not None:
                    if upper - lower <= self.tolerance:
                        logging.info("Tolerance reached; storing %.4f.", lower)
                        self._store_result(idx, lower)
                        break
                    next_current = max(self.min_step, (lower + upper) / 2.0)
                else:
                    next_current = max(self.min_step, current * self.decrease_factor)

            elif response in {"n", "no"}:
                lower = current
                direction = "up"
                if upper is not None:
                    if upper - lower <= self.tolerance:
                        logging.info("Tolerance reached; storing %.4f.", lower)
                        self._store_result(idx, lower)
                        break
                    next_current = min(self.max_step, (lower + upper) / 2.0)
                else:
                    next_current = min(self.max_step, current * self.increase_factor)

            elif response in {"a", "accept"}:
                accepted = lower if lower is not None else current
                self._store_result(idx, accepted)
                break

            elif response in {"b", "set"}:
                value_str = input("Enter amplitude value: ").strip()
                try:
                    candidate = float(value_str)
                except ValueError:
                    logging.warning("Invalid amplitude '%s'.", value_str)
                    continue
                current = float(np.clip(candidate, self.min_step, self.max_step))
                need_play = True
                continue

            elif response in {"r", "repeat"}:
                need_play = True
                continue

            elif response in {"s", "skip"}:
                if self.results[idx] is not None:
                    logging.info("Keeping stored amplitude %.4f.", self.results[idx])
                    self._store_result(idx, self.results[idx])
                else:
                    logging.info("No previous value for subband %d; leaving unchanged.", idx)
                break

            elif response in {"q", "quit"}:
                raise KeyboardInterrupt

            else:
                logging.info("Please answer with y, n, a, b, r, s, or q.")
                continue

            if last_direction and direction != last_direction:
                reversals += 1
                if lower is not None and upper is not None and reversals >= self.target_reversals:
                    logging.info(
                        "Reached %d reversals; storing lower bound %.4f.",
                        self.target_reversals,
                        lower,
                    )
                    self._store_result(idx, lower)
                    break

            current = next_current
            last_direction = direction
            need_play = True

    def _store_result(self, idx: int, value: float) -> None:
        info = self.subbands[idx]
        self.results[idx] = value
        self.base_coeffs[info.slc, :] = self.templates[idx] * value
        logging.info("Stored ToH %.4f for subband %d.", value, idx)

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------
    def _play_trial(self, subband_idx: int, amplitude: float) -> None:
        coeffs = self.base_coeffs.copy()
        coeffs[self.subbands[subband_idx].slc, :] = self.templates[subband_idx] * amplitude
        chunk = self._coeffs_to_time(coeffs)
        noise_segment = self._repeat_to_length(chunk, self.noise_seconds)
        silence_frames = max(1, int(round(self.silence_seconds * self.sample_rate)))
        silence_segment = np.zeros((silence_frames, self.channels), dtype=np.float64)

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
        sd.play(audio, samplerate=int(self.sample_rate))
        sd.wait()

    def _coeffs_to_time(self, coeff_array: np.ndarray) -> np.ndarray:
        chunk = np.empty((self.frames_per_chunk, self.channels), dtype=np.float64)
        for ch in range(self.channels):
            coeffs = pywt.array_to_coeffs(
                coeff_array[:, ch],
                self.slices,
                output_format="wavedec",
            )
            chunk[:, ch] = pywt.waverec(coeffs, wavelet=self.wavelet, mode="per")
        return chunk

    def _repeat_to_length(self, chunk: np.ndarray, seconds: float) -> np.ndarray:
        target_frames = max(1, int(round(seconds * self.sample_rate)))
        repeats = max(1, int(math.ceil(target_frames / chunk.shape[0])))
        tiled = np.tile(chunk, (repeats, 1))
        return tiled[:target_frames]


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
        session = CustomToHSession()
        session.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nCalibration interrupted by user.")
    except RuntimeError as exc:
        logging.error("%s", exc)
        minimal.parser.exit(1)
