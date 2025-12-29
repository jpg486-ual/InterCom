#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""Dyadic ToH variant using linear subband decomposition based on Wavelet Packets."""

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pywt

import minimal
from temporal_overlapped_DWT_coding import (
    Temporal_Overlapped_DWT,
    Temporal_Overlapped_DWT__verbose,
)
from DEFLATE_byteplanes3 import DEFLATE_BytePlanes3 as EC


minimal.parser.add_argument(
    "--linear_subbands",
    type=int,
    default=2,
    help="Number of Wavelet Packet subbands per dyadic subband (power of two).",
)
minimal.parser.add_argument(
    "--linear_max_q",
    type=int,
    default=128,
    help="Maximum quantization multiplier for linear subbands.",
)
minimal.parser.add_argument(
    "--linear_wavelet_name",
    type=str,
    default="db4",
    help="Wavelet name used for the Wavelet Packet decomposition (default: 'db4').",
)
minimal.parser.add_argument(
    "--custom_ToH",
    action="store_true",
    help="Load custom ToH thresholds from ./custom_ToH.txt for dyadic bands.",
)

@dataclass
class _SubbandInfo:
    idx_slice: slice
    length: int
    freq_start: float
    freq_end: float
    label: str


class Dyadic_Linear_ToH(Temporal_Overlapped_DWT):
    """Temporal_Overlapped_DWT with dyadic + linear subband quantization."""

    def __init__(self) -> None:
        super().__init__()
        self.linear_subbands = int(max(1, minimal.args.linear_subbands))
        if self.linear_subbands & (self.linear_subbands - 1):
            raise ValueError("--linear_subbands must be a positive power of two")
        self.wpt_levels = int(math.log2(self.linear_subbands)) if self.linear_subbands > 1 else 0

        wpt_wavelet_name = minimal.args.linear_wavelet_name or minimal.args.wavelet_name
        try:
            self.wpt_wavelet = pywt.Wavelet(wpt_wavelet_name)
        except ValueError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"Unknown wavelet for WPT: '{wpt_wavelet_name}'") from exc
        self.wpt_filter_length = self.wpt_wavelet.dec_len

        self._coef_index = np.arange(minimal.args.frames_per_chunk, dtype=np.int32)
        self._subbands: List[_SubbandInfo] = self._build_subband_metadata()
        self._packet_slices = self._build_packet_slices()
        self._custom_toh = self._load_custom_toh() if minimal.args.custom_ToH else None
        self.quantization_steps = self._compute_quantization_steps(max_q=max(1, minimal.args.linear_max_q))
        self._active_steps = self.quantization_steps.astype(np.int32, copy=True)
        logging.info(
            "linear subbands per band = %d, wpt levels = %d",
            self.linear_subbands,
            self.wpt_levels,
        )
        logging.info("total linear subbands = %d", len(self.quantization_steps))
        logging.info("wavelet packet wavelet = %s", self.wpt_wavelet.name)

    def calc(self, frequency: float) -> float:
        """Threshold of human hearing formula (20-year-old reference)."""
        f_khz = frequency / 1000.0
        return (
            3.64 * (f_khz ** -0.8)
            - 6.5 * math.exp(-0.6 * (f_khz - 3.3) ** 2)
            + (10 ** -3) * (f_khz ** 4)
        )

    def _load_custom_toh(self) -> List[float] | None:
        table_path = Path(__file__).with_name("custom_ToH.txt")
        if not table_path.exists():
            logging.warning("custom_ToH flag enabled but %s not found. Using fallback thresholds.", table_path)
            return None

        values: List[float] = []
        try:
            with table_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.split('#')[0].strip()
                    if not line or line.startswith("["): continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            values.append(float(parts[2]))
                        except ValueError:
                            logging.warning("Invalid ToH value '%s' in %s.", parts[2], table_path)
                            continue
        except OSError as exc:
            logging.warning("Unable to read %s (%s). Using fallback thresholds.", table_path, exc)
            return None

        expected = 1 + self.dwt_levels
        if len(values) != expected:
            logging.warning(
                "custom_ToH.txt contains %d entries but %d are required (levels + 1). Using fallback thresholds.",
                len(values),
                expected,
            )
            return None

        logging.info("Loaded %d custom ToH values from %s", len(values), table_path)
        return values

    def _build_subband_metadata(self) -> List[_SubbandInfo]:
        nyquist = minimal.args.frames_per_second / 2.0
        subbands: List[_SubbandInfo] = []

        approx_slice = self.slices[0][0]
        approx_len = self._slice_length(approx_slice)
        self._validate_length(approx_len, "approximation")
        approx_end = nyquist / (2 ** self.dwt_levels)
        subbands.append(
            _SubbandInfo(
                idx_slice=approx_slice,
                length=approx_len,
                freq_start=0.0,
                freq_end=approx_end,
                label="A",
            )
        )

        for i in range(self.dwt_levels):
            detail_slice = self.slices[i + 1]["d"][0]
            detail_len = self._slice_length(detail_slice)
            detail_level = self.dwt_levels - i
            start = nyquist / (2 ** detail_level)
            end = nyquist if detail_level == 1 else nyquist / (2 ** (detail_level - 1))
            self._validate_length(detail_len, f"detail L{detail_level}")
            subbands.append(
                _SubbandInfo(
                    idx_slice=detail_slice,
                    length=detail_len,
                    freq_start=start,
                    freq_end=end,
                    label=f"D{detail_level}",
                )
            )

        return subbands

    def _validate_length(self, length: int, label: str) -> None:
        if length % self.linear_subbands != 0:
            raise ValueError(
                f"Subband '{label}' of length {length} cannot be split into "
                f"{self.linear_subbands} packets"
            )
        if self.wpt_levels:
            min_required = 1 << self.wpt_levels
            if length < min_required:
                raise ValueError(
                    f"Subband '{label}' length {length} shorter than required "
                    f"{min_required} samples for WPT"
                )
            max_level = pywt.dwt_max_level(data_len=length, filter_len=self.wpt_filter_length)
            if self.wpt_levels > max_level:
                raise ValueError(
                    f"Subband '{label}' only supports {max_level} WPT levels with the "
                    f"selected WPT wavelet"
                )

    def _slice_length(self, slc: slice) -> int:
        return int(self._coef_index[slc].size)

    def _build_packet_slices(self) -> List[List[slice]]:
        slices: List[List[slice]] = []
        offset = 0
        for info in self._subbands:
            segment = []
            packet_len = info.length // self.linear_subbands
            for _ in range(self.linear_subbands):
                segment.append(slice(offset, offset + packet_len))
                offset += packet_len
            slices.append(segment)
        expected = minimal.args.frames_per_chunk
        if offset != expected:
            raise ValueError(
                f"Linear packet layout mismatch: expected {expected}, got {offset}"
            )
        return slices

    def _compute_quantization_steps(self, max_q: int) -> np.ndarray:
        averages = []
        for info in self._subbands:
            width = info.freq_end - info.freq_start
            sub_width = width / self.linear_subbands if self.linear_subbands else width
            for packet_idx in range(self.linear_subbands):
                start = info.freq_start + packet_idx * sub_width
                end = start + sub_width
                averages.append(self._average_spl(start, end))
        if not averages:
            return np.array([], dtype=np.int32)
        if self._custom_toh is not None:
            base_thresholds = np.repeat(np.asarray(self._custom_toh, dtype=np.float64), self.linear_subbands)
            if base_thresholds.size != len(averages):
                logging.warning("Custom ToH size mismatch after expansion. Using fallback thresholds.")
                self._custom_toh = None
                return self._compute_quantization_steps(max_q)
            baseline = base_thresholds
        else:
            baseline = np.asarray(averages, dtype=np.float64)
        min_spl = float(baseline.min())
        max_spl = float(baseline.max())
        quant_steps = []
        for i, avg in enumerate(averages):
            if max_spl == min_spl:
                scaled = 0.0
            else:
                scaled = (baseline[i] - min_spl) / (max_spl - min_spl)
            factor = scaled * (max_q - 1) + 1
            step = max(round(factor * minimal.args.minimal_quantization_step_size), 1)
            quant_steps.append(max(step, minimal.args.minimal_quantization_step_size))
        result = np.asarray(quant_steps, dtype=np.int32)
        expected = len(self._subbands) * self.linear_subbands
        if result.size != expected:
            raise RuntimeError("Quantization step table size mismatch")
        logging.info("Quantization step sizes: %s", result.tolist())
        return result

    def _average_spl(self, start: float, end: float) -> float:
        # Use logarithmic weighting to emphasize lower frequencies
        start = max(start, 1.0)
        end = max(end, start + 1e-6)
        samples = max(16, int((end - start) // 50) * 16)
        frequencies = np.linspace(start, end, samples, endpoint=False)
        values = [self.calc(freq) for freq in frequencies]

        # Weight lower frequencies more strongly
        weights = 1 / np.sqrt(frequencies)
        weighted_avg = np.sum(np.array(values) * weights) / np.sum(weights)
        return float(weighted_avg)

    def _apply_wpt(self, data: np.ndarray) -> List[np.ndarray]:
        if self.wpt_levels == 0:
            return [data.astype(np.float64, copy=False)]
        wp = pywt.WaveletPacket(data=data, wavelet=self.wpt_wavelet, maxlevel=self.wpt_levels, mode="per")
        return [node.data.copy() for node in wp.get_level(self.wpt_levels, order="natural")]

    def _reconstruct_from_packets(self, packets: List[np.ndarray], length: int) -> np.ndarray:
        if self.wpt_levels == 0:
            return packets[0].astype(np.float64, copy=False)
        wp = pywt.WaveletPacket(
            data=np.zeros(length, dtype=np.float64),
            wavelet=self.wpt_wavelet,
            maxlevel=self.wpt_levels,
            mode="per",
        )
        for node, data in zip(wp.get_level(self.wpt_levels, order="natural"), packets):
            node.data = data
        return wp.reconstruct(update=False)

    def pack(self, chunk_number: int, chunk: np.ndarray) -> bytes:
        dwt_chunk = super().analyze(chunk)
        packets = np.empty_like(dwt_chunk)
        steps = self._active_steps
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
                    packet_idx += 1
            if packet_idx != len(steps):
                raise RuntimeError("Unexpected packet count during packing")
        packets = packets.astype(np.int32, copy=False)
        return EC.pack(self, chunk_number, packets)

    def unpack(self, packed_chunk: bytes):
        chunk_number, packets = EC.unpack(self, packed_chunk)
        dwt_chunk = np.empty_like(packets, dtype=np.float64)
        steps = self._active_steps
        for channel in range(minimal.args.number_of_channels):
            packet_idx = 0
            for info, slices in zip(self._subbands, self._packet_slices):
                packet_list = []
                for target_slice in slices:
                    current_idx = packet_idx
                    step = steps[current_idx]
                    data = packets[target_slice, channel].astype(np.float64, copy=False) * step
                    packet_list.append(data)
                    packet_idx += 1
                reconstructed = self._reconstruct_from_packets(packet_list, info.length)
                dwt_chunk[info.idx_slice, channel] = reconstructed
            if packet_idx != len(steps):
                raise RuntimeError("Unexpected packet count during unpacking")
        dwt_chunk = dwt_chunk.astype(np.int32, copy=False)
        chunk = super().synthesize(dwt_chunk)
        return chunk_number, chunk


class Dyadic_Linear_ToH__verbose(Dyadic_Linear_ToH, Temporal_Overlapped_DWT__verbose):
    pass


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
        intercom = Dyadic_Linear_ToH__verbose()
    else:
        intercom = Dyadic_Linear_ToH()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()
