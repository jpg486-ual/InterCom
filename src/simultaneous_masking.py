#!/usr/bin/env python
"""Simultaneous masking codec built on Dyadic_Linear_ToH.

This module follows the guidelines described in
https://tecnologias-multimedia.github.io/contents/simultaneous_masking/ and
implements the masking controller as a subclass of the dyadic linear ToH codec.
The controller measures per-packet energies, applies the neighbour factors, and
updates the quantization steps in a smoothed manner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

import minimal
from DEFLATE_byteplanes3 import DEFLATE_BytePlanes3 as EC
from dyadic_linear_ToH import Dyadic_Linear_ToH, Dyadic_Linear_ToH__verbose


minimal.parser.add_argument(
    "--masking_smoothing",
    type=float,
    default=0.6,
    help="Exponential smoothing factor for masking updates (0 disables smoothing).",
)
minimal.parser.add_argument(
    "--masking_factors",
    type=str,
    default="1.0,1.5,2.0",
    help="Comma-separated neighbour factors applied around the masker band.",
)


def _parse_masking_factors(raw: str) -> tuple[float, ...]:
    try:
        parts = [float(item) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--masking_factors must be a comma-separated list of floats") from exc
    if not parts:
        raise ValueError("--masking_factors must provide at least one value")
    return tuple(parts)


@dataclass(frozen=True)
class MaskingConfig:
    """Configuration parameters for the masking controller.

    Attributes
    ----------
    minimal_step:
        Smallest allowed quantization step (matches codec minimum).
    maximal_step:
        Largest allowed quantization step (matches codec maximum).
    smoothing_factor:
        Exponential moving average factor in [0, 1). Higher means slower
        response but better temporal stability.
    neighbour_factors:
        Multiplicative factors applied to the masker band (index 0) and its
        neighbours (distance-based). Default implements the pattern described in
        the lecture notes: [1.0, 2.0, 3.0]. Distances longer than the sequence
        length clamp to the last factor.
    energy_floor:
        Small positive constant to avoid numerical issues when energies are
        extremely small (silence).
    """

    minimal_step: int
    maximal_step: int
    smoothing_factor: float = 0.6
    neighbour_factors: Sequence[float] = (1.0, 2.0, 3.0)
    energy_floor: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.smoothing_factor < 1.0:
            raise ValueError("smoothing_factor must be in [0, 1)")
        if self.minimal_step <= 0:
            raise ValueError("minimal_step must be positive")
        if self.maximal_step < self.minimal_step:
            raise ValueError("maximal_step must be >= minimal_step")
        if len(self.neighbour_factors) == 0:
            raise ValueError("neighbour_factors cannot be empty")
        if self.neighbour_factors[0] != 1.0:
            raise ValueError("neighbour_factors[0] must be 1.0 for the masker band")
        if any(f <= 0 for f in self.neighbour_factors):
            raise ValueError("neighbour_factors must contain positive values")
        if self.energy_floor <= 0.0:
            raise ValueError("energy_floor must be positive")


class MaskingController:
    """Stateful controller that adjusts quantization steps using masking."""

    def __init__(self, config: MaskingConfig) -> None:
        self._config = config
        self._previous_steps: Optional[np.ndarray] = None

    @property
    def config(self) -> MaskingConfig:
        return self._config

    def reset(self) -> None:
        """Clears the temporal smoothing state."""

        self._previous_steps = None

    def compute_energies(self, packets: Iterable[np.ndarray]) -> np.ndarray:
        """Returns the energy of each subband packet.

        Parameters
        ----------
        packets:
            Iterable of NumPy arrays (one per subband) holding real-valued
            coefficients.
        """

        energies = [float(np.sum(np.square(np.asarray(packet, dtype=np.float64)))) for packet in packets]
        if not energies:
            raise ValueError("packets cannot be empty")
        return np.maximum(np.asarray(energies, dtype=np.float64), self._config.energy_floor)

    def adjust_steps(self, energies: Sequence[float], base_steps: Sequence[int]) -> np.ndarray:
        """Computes masked quantization steps.

        Parameters
        ----------
        energies:
            Energy values per subband (one per linear subband).
        base_steps:
            Baseline quantization steps matching the codec configuration.
        """

        base = np.asarray(base_steps, dtype=np.float64)
        if base.ndim != 1:
            raise ValueError("base_steps must be one-dimensional")
        energy = np.asarray(energies, dtype=np.float64)
        if energy.ndim != 1:
            raise ValueError("energies must be one-dimensional")
        if base.size != energy.size:
            raise ValueError("energies and base_steps must have the same length")
        if base.size == 0:
            raise ValueError("inputs cannot be empty")

        energy = np.maximum(energy, self._config.energy_floor)
        masker_index = int(np.argmax(energy))
        factors = self._build_factors(masker_index, base.size)

        raw_steps = np.clip(base * factors, self._config.minimal_step, self._config.maximal_step)
        if self._previous_steps is None:
            smoothed = raw_steps
        else:
            alpha = self._config.smoothing_factor
            smoothed = alpha * self._previous_steps + (1.0 - alpha) * raw_steps
        smoothed = np.clip(smoothed, self._config.minimal_step, self._config.maximal_step)
        self._previous_steps = smoothed
        return smoothed.astype(np.int32, copy=False)

    def _build_factors(self, masker_index: int, length: int) -> np.ndarray:
        """Returns multiplicative factors following the neighbour pattern."""

        distances = np.abs(np.arange(length) - masker_index)
        last_idx = len(self._config.neighbour_factors) - 1
        factors = np.empty(length, dtype=np.float64)
        for i, distance in enumerate(distances):
            idx = int(distance)
            if idx > last_idx:
                idx = last_idx
            factors[i] = self._config.neighbour_factors[idx]
        return factors

    def snapshot(self) -> Optional[np.ndarray]:
        """Exposes the latest smoothed steps (useful for telemetry)."""

        return None if self._previous_steps is None else self._previous_steps.copy()


class SimultaneousMasking(Dyadic_Linear_ToH):
    """Dyadic_Linear_ToH variant with simultaneous masking enabled."""

    def __init__(self) -> None:
        super().__init__()
        max_step = minimal.args.linear_max_q * minimal.args.minimal_quantization_step_size
        factors = _parse_masking_factors(minimal.args.masking_factors)
        self._masking = MaskingController(
            MaskingConfig(
                minimal_step=minimal.args.minimal_quantization_step_size,
                maximal_step=max_step,
                smoothing_factor=minimal.args.masking_smoothing,
                neighbour_factors=factors,
            )
        )
        logging.info(
            "simultaneous masking enabled (smoothing=%s, factors=%s, max_step=%s)",
            minimal.args.masking_smoothing,
            list(factors),
            max_step,
        )

    def _update_masking_steps(self, energies: np.ndarray) -> None:
        try:
            self._active_steps = self._masking.adjust_steps(energies, self.quantization_steps)
        except ValueError as exc:
            logging.warning("Masking adjustment skipped (%s)", exc)

    def pack(self, chunk_number: int, chunk: np.ndarray) -> bytes:
        dwt_chunk = super().analyze(chunk)
        packets = np.empty_like(dwt_chunk)
        steps = self._active_steps
        energy_accumulator = np.zeros_like(steps, dtype=np.float64)
        for channel in range(minimal.args.number_of_channels):
            packet_idx = 0
            for info, slices in zip(self._subbands, self._packet_slices):
                subband_data = dwt_chunk[info.idx_slice, channel].astype(np.float64, copy=False)
                wpt_packets = self._apply_wpt(subband_data)
                for node_data, target_slice in zip(wpt_packets, slices):
                    current_idx = packet_idx
                    step = steps[current_idx]
                    quantized = np.rint(node_data / step).astype(np.int32)
                    packets[target_slice, channel] = quantized
                    reconstructed = quantized.astype(np.float64, copy=False) * step
                    energy_accumulator[current_idx] += float(np.sum(reconstructed * reconstructed))
                    packet_idx += 1
            if packet_idx != len(steps):
                raise RuntimeError("Unexpected packet count during packing")
        self._update_masking_steps(energy_accumulator)
        packets = packets.astype(np.int32, copy=False)
        return EC.pack(self, chunk_number, packets)

    def unpack(self, packed_chunk: bytes):
        chunk_number, packets = EC.unpack(self, packed_chunk)
        dwt_chunk = np.empty_like(packets, dtype=np.float64)
        steps = self._active_steps
        energy_accumulator = np.zeros_like(steps, dtype=np.float64)
        for channel in range(minimal.args.number_of_channels):
            packet_idx = 0
            for info, slices in zip(self._subbands, self._packet_slices):
                packet_list = []
                for target_slice in slices:
                    current_idx = packet_idx
                    step = steps[current_idx]
                    data = packets[target_slice, channel].astype(np.float64, copy=False) * step
                    energy_accumulator[current_idx] += float(np.sum(data * data))
                    packet_list.append(data)
                    packet_idx += 1
                reconstructed = self._reconstruct_from_packets(packet_list, info.length)
                dwt_chunk[info.idx_slice, channel] = reconstructed
            if packet_idx != len(steps):
                raise RuntimeError("Unexpected packet count during unpacking")
        dwt_chunk = dwt_chunk.astype(np.int32, copy=False)
        chunk = super().synthesize(dwt_chunk)
        self._update_masking_steps(energy_accumulator)
        return chunk_number, chunk


class SimultaneousMasking__verbose(SimultaneousMasking, Dyadic_Linear_ToH__verbose):
    pass


__all__ = [
    "MaskingConfig",
    "MaskingController",
    "SimultaneousMasking",
    "SimultaneousMasking__verbose",
]


try:
    import argcomplete  # type: ignore
except ImportError:  # pragma: no cover
    logging.warning("Unable to import argcomplete (optional)")


if __name__ == "__main__":
    minimal.parser.description = __doc__
    try:  # pragma: no cover
        argcomplete.autocomplete(minimal.parser)  # type: ignore
    except Exception:  # pragma: no cover
        logging.warning("argcomplete not working :-/")

    minimal.args = minimal.parser.parse_known_args()[0]

    if minimal.args.show_stats or minimal.args.show_samples or minimal.args.show_spectrum:
        intercom = SimultaneousMasking__verbose()
    else:
        intercom = SimultaneousMasking()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()
