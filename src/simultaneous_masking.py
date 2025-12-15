#!/usr/bin/env python
"""Simultaneous masking controller for dynamic quantization step sizing.

This module provides a light-weight implementation of the guidelines described in
https://tecnologias-multimedia.github.io/contents/simultaneous_masking/.

It computes per-chunk masking gains from subband energies and adjusts the
baseline quantization step sizes accordingly. The design keeps the focus on the
InterCom use-case, where encoder and decoder share the same baseline
configuration and the controller emits bounded, smoothed adjustments that can be
transmitted or re-generated deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


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


class SimultaneousMasking:
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


__all__ = ["MaskingConfig", "SimultaneousMasking"]
