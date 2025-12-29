#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""Feedback cancellation using frequency-domain subtraction with live controls."""

import logging
import math
import sys
import threading
from collections import deque

import numpy as np

import minimal
import buffer

minimal.parser.add_argument(
    "--feedback_attenuation",
    type=float,
    default=0.9,
    help="Attenuation scalar 'a' applied to the playback spectrum before subtraction",
)
minimal.parser.add_argument(
    "--feedback_min_att",
    type=float,
    default=0.0,
    help="Lower bound for the attenuation scalar when tuning it interactively",
)
minimal.parser.add_argument(
    "--feedback_max_att",
    type=float,
    default=1.5,
    help="Upper bound for the attenuation scalar when tuning it interactively",
)
minimal.parser.add_argument(
    "--feedback_step",
    type=float,
    default=0.05,
    help="Step used to modify the attenuation with the keyboard controls",
)
minimal.parser.add_argument(
    "--feedback_delay_ms",
    type=float,
    default=0.0,
    help="Initial estimate of the acoustic delay between speakers and microphone (milliseconds)",
)
minimal.parser.add_argument(
    "--feedback_delay_step_ms",
    type=float,
    default=1.0,
    help="Step used to modify the delay with the keyboard controls (milliseconds)",
)
minimal.parser.add_argument(
    "--feedback_max_delay_ms",
    type=float,
    default=300.0,
    help="Upper bound for the delay slider (milliseconds)",
)
minimal.parser.add_argument(
    "--feedback_auto_calibrate",
    action="store_true",
    help="Play a calibration tone on startup to estimate attenuation and delay automatically",
)
minimal.parser.add_argument(
    "--feedback_calibration_freq",
    type=float,
    default=1000.0,
    help="Frequency of the calibration tone in Hertz",
)
minimal.parser.add_argument(
    "--feedback_calibration_duration",
    type=float,
    default=2.0,
    help="Duration of the calibration tone in seconds",
)
minimal.parser.add_argument(
    "--feedback_calibration_guard_chunks",
    type=int,
    default=2,
    help="Number of extra chunks to record after the tone to capture its decay",
)
minimal.parser.add_argument(
    "--feedback_no_gui",
    action="store_true",
    help="Disable Matplotlib sliders for attenuation and delay (CLI controls only)",
)


class Feedback_Cancellation(buffer.Buffering):
    def __init__(self):
        super().__init__()
        logging.info(__doc__)
        self._attenuation_min = minimal.args.feedback_min_att
        self._attenuation_max = max(
            minimal.args.feedback_min_att,
            minimal.args.feedback_max_att,
        )
        self._attenuation_step = max(minimal.args.feedback_step, 0.0)
        self._control_help_text = (
            "Controls: '+', '=', or ']' increase attenuation; '-', '_' or '[' decrease attenuation; 'a <value>' sets attenuation; '>' increases delay; '<' decreases delay; 'd <ms>' sets delay milliseconds; '0' resets delay to 0; 'h' prints help."
        )
        self.attenuation = self._clamp_attenuation(minimal.args.feedback_attenuation)
        self.delay_samples = max(
            0.0,
            minimal.args.feedback_delay_ms * minimal.args.frames_per_second / 1000.0,
        )
        self._delay_step_ms = max(minimal.args.feedback_delay_step_ms, 0.0)
        if minimal.args.frames_per_chunk > 0:
            estimated_history = int(
                math.ceil(
                    (minimal.args.frames_per_second * 0.5)
                    / minimal.args.frames_per_chunk
                )
            )
        else:
            estimated_history = 8
        history_length = max(8, estimated_history)
        self._play_history = deque(maxlen=history_length)
        self._last_sent_chunk = self.generate_zero_chunk()
        self._reference_chunk = self.generate_zero_chunk()
        self._calibrating = minimal.args.feedback_auto_calibrate
        self._calibration_guard = max(minimal.args.feedback_calibration_guard_chunks, 0)
        self._calibration_tone_freq = max(minimal.args.feedback_calibration_freq, 20.0)
        self._calibration_tone_phase = 0.0
        self._calibration_played = []
        self._calibration_recorded = []
        self._calibration_started = False
        self._calibration_completed = False
        calibration_duration = max(minimal.args.feedback_calibration_duration, 0.0)
        self._calibration_chunks_left = 0
        self._calibration_recordings_needed = 0
        if self._calibrating:
            self._calibration_chunks_left = max(
                1,
                int(math.ceil(calibration_duration / self.chunk_time)),
            )
            self._calibration_recordings_needed = (
                self._calibration_chunks_left + self._calibration_guard + 2
            )
            logging.info(
                "Calibration tone enabled: %d chunks at %.1f Hz",
                self._calibration_chunks_left,
                self._calibration_tone_freq,
            )
        else:
            self._calibration_guard = 0
        self._slider_internal_update = False
        self._attenuation_slider = None
        self._delay_slider = None
        self._slider_fig = None
        self._control_thread = None
        if sys.stdin.isatty():
            self._start_control_thread()
        else:
            logging.info("Interactive feedback controls disabled (stdin not a TTY)")
        if not minimal.args.feedback_no_gui:
            self._start_slider_ui()
        logging.info(
            "Initial feedback params -> attenuation: %.3f, delay: %.2f ms",
            self.attenuation,
            self.delay_ms,
        )

    def _clamp_attenuation(self, value):
        return float(np.clip(value, self._attenuation_min, self._attenuation_max))

    def _start_control_thread(self):
        if self._attenuation_step == 0.0:
            logging.info("feedback_step is 0. Use 'a <value>' to update the attenuation.")
        logging.info(
            "Interactive feedback controls ready. %s Current attenuation: %.3f",
            self._control_help_text,
            self.attenuation,
        )
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="FeedbackControls",
            daemon=True,
        )
        self._control_thread.start()

    @property
    def delay_ms(self):
        if minimal.args.frames_per_second <= 0:
            return 0.0
        return self.delay_samples * 1000.0 / minimal.args.frames_per_second

    def set_delay_samples(self, samples):
        samples = max(0.0, float(samples))
        self.delay_samples = samples
        logging.info(
            "feedback delay set to %.2f ms (%.1f samples)",
            self.delay_ms,
            self.delay_samples,
        )
        self._update_slider_value(self._delay_slider, self.delay_ms)

    def set_delay_ms(self, value_ms):
        value_ms = max(0.0, float(value_ms))
        samples = value_ms * minimal.args.frames_per_second / 1000.0
        self.set_delay_samples(samples)

    def adjust_delay_ms(self, delta_ms):
        self.set_delay_ms(self.delay_ms + delta_ms)

    def _control_loop(self):
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break
            command = line.strip().lower()
            if not command:
                continue
            if command in {"+", "=", "]"}:
                self.adjust_attenuation(self._attenuation_step)
            elif command in {"-", "_", "["}:
                self.adjust_attenuation(-self._attenuation_step)
            elif command == ">":
                if self._delay_step_ms == 0.0:
                    logging.info(
                        "feedback_delay_step_ms is 0. Use 'd <ms>' to set the delay."
                    )
                else:
                    self.adjust_delay_ms(self._delay_step_ms)
            elif command == "<":
                if self._delay_step_ms == 0.0:
                    logging.info(
                        "feedback_delay_step_ms is 0. Use 'd <ms>' to set the delay."
                    )
                else:
                    self.adjust_delay_ms(-self._delay_step_ms)
            elif command.startswith("a"):
                parts = command.split()
                if len(parts) == 2:
                    try:
                        value = float(parts[1])
                    except ValueError:
                        logging.warning("Invalid attenuation value: %s", parts[1])
                        continue
                    self.set_attenuation(value)
                else:
                    logging.info("Usage: a <value>")
            elif command.startswith("d"):
                parts = command.split()
                if len(parts) == 2:
                    try:
                        value_ms = float(parts[1])
                    except ValueError:
                        logging.warning("Invalid delay value (ms): %s", parts[1])
                        continue
                    self.set_delay_ms(value_ms)
                else:
                    logging.info("Usage: d <milliseconds>")
            elif command == "0":
                self.set_delay_ms(0.0)
            elif command in {"h", "help"}:
                logging.info(self._control_help_text)
            else:
                logging.info("Unknown command '%s'. %s", command, self._control_help_text)

    def set_attenuation(self, value):
        self.attenuation = self._clamp_attenuation(value)
        logging.info("feedback attenuation set to %.3f", self.attenuation)
        self._update_slider_value(self._attenuation_slider, self.attenuation)

    def adjust_attenuation(self, delta):
        self.set_attenuation(self.attenuation + delta)

    def _store_playback(self, chunk_matrix):
        self._play_history.append(chunk_matrix.copy())

    def _get_reference_playback(self):
        if not self._play_history:
            return self.zero_chunk

        chunk_size = minimal.args.frames_per_chunk
        channels = minimal.args.number_of_channels
        if chunk_size <= 0 or channels <= 0:
            return self.zero_chunk

        delay = float(self.delay_samples)
        if delay <= 0.0:
            return self._play_history[-1]

        history_matrix = np.concatenate(self._play_history, axis=0).astype(
            np.float32,
            copy=False,
        )
        total_samples = history_matrix.shape[0]

        if total_samples < 2:
            return self.zero_chunk

        if delay >= total_samples - 1:
            return self.zero_chunk

        start = total_samples - delay - chunk_size
        if start < 0:
            return self.zero_chunk

        sample_positions = start + np.arange(chunk_size, dtype=np.float64)
        lower_indexes = np.floor(sample_positions).astype(np.int64)
        upper_indexes = np.clip(lower_indexes + 1, 0, total_samples - 1)
        alphas = (sample_positions - lower_indexes)[:, np.newaxis]

        reference = (
            (1.0 - alphas) * history_matrix[lower_indexes]
            + alphas * history_matrix[upper_indexes]
        )

        return np.clip(np.round(reference), -32768, 32767).astype(np.int16)

    def _maybe_override_playback(self, playback_matrix):
        if not self._calibrating or self._calibration_chunks_left <= 0:
            return playback_matrix, playback_matrix.reshape(-1)
        if not self._calibration_started:
            self._calibration_started = True
            logging.info(
                "Calibration tone started (%d chunks remaining)",
                self._calibration_chunks_left,
            )
        tone_chunk = self._generate_calibration_chunk()
        self._calibration_played.append(tone_chunk.copy())
        self._calibration_chunks_left -= 1
        return tone_chunk, tone_chunk.reshape(-1)

    def _generate_calibration_chunk(self):
        frames = minimal.args.frames_per_chunk
        channels = minimal.args.number_of_channels
        if frames <= 0 or channels <= 0:
            return self.zero_chunk
        omega = 2.0 * np.pi * self._calibration_tone_freq / minimal.args.frames_per_second
        sample_indices = np.arange(frames, dtype=np.float64)
        phase = self._calibration_tone_phase + omega * sample_indices
        self._calibration_tone_phase = math.fmod(
            self._calibration_tone_phase + omega * frames,
            2.0 * np.pi,
        )
        tone = np.sin(phase, dtype=np.float64)
        amplitude = 0.6 * 32767.0
        tone = np.clip(tone * amplitude, -32767.0, 32767.0).astype(np.int16)
        if channels == 1:
            tone = tone.reshape(-1, 1)
        else:
            tone = np.tile(tone.reshape(-1, 1), (1, channels))
        return tone

    def _maybe_store_calibration_recording(self, recorded_chunk):
        if not self._calibrating or not self._calibration_started:
            return
        self._calibration_recorded.append(recorded_chunk.copy())
        if (
            not self._calibration_completed
            and self._calibration_chunks_left == 0
            and len(self._calibration_recorded) >= self._calibration_recordings_needed
        ):
            self._finish_calibration()

    def _finish_calibration(self):
        if not self._calibration_played or not self._calibration_recorded:
            logging.warning("Calibration did not capture enough data")
            self._calibrating = False
            self._calibration_completed = True
            return
        played = np.concatenate(self._calibration_played, axis=0).astype(np.float64)
        recorded = np.concatenate(self._calibration_recorded, axis=0).astype(np.float64)
        played_mono = played.mean(axis=1)
        recorded_mono = recorded.mean(axis=1)
        played_mono -= np.mean(played_mono)
        recorded_mono -= np.mean(recorded_mono)
        played_rms = np.sqrt(np.mean(played_mono ** 2))
        recorded_rms = np.sqrt(np.mean(recorded_mono ** 2))
        if played_rms == 0 or recorded_rms == 0:
            logging.warning("Calibration tone produced zero energy; skipping adjustments")
        else:
            corr = np.correlate(recorded_mono, played_mono, mode="full")
            lag = np.argmax(corr) - (played_mono.size - 1)
            if lag < 0:
                lag = 0
            self.set_delay_samples(lag)
            attenuation = recorded_rms / played_rms
            self.set_attenuation(attenuation)
            logging.info(
                "Calibration finished: delay %.2f ms, attenuation %.3f",
                self.delay_ms,
                self.attenuation,
            )
        self._calibrating = False
        self._calibration_completed = True
        self._calibration_played.clear()
        self._calibration_recorded.clear()

    def cancel_feedback(self, recorded_chunk, played_chunk):
        """Apply N^ = M - a S in the frequency domain (Eq. (15))."""
        recorded = recorded_chunk.astype(np.float32, copy=False)
        played = played_chunk.astype(np.float32, copy=False)

        recorded_fft = np.fft.rfft(recorded, axis=0)
        played_fft = np.fft.rfft(played, axis=0)

        cancelled_fft = recorded_fft - self.attenuation * played_fft
        cancelled = np.fft.irfft(
            cancelled_fft,
            n=recorded_chunk.shape[0],
            axis=0,
        )

        cancelled = np.clip(np.round(cancelled), -32768, 32767).astype(np.int16)
        return cancelled

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        self.chunk_number = (self.chunk_number + 1) % self.CHUNK_NUMBERS

        playback_chunk = self.unbuffer_next_chunk()
        playback_matrix = playback_chunk.reshape(
            minimal.args.frames_per_chunk, minimal.args.number_of_channels
        )
        playback_matrix, playback_chunk = self._maybe_override_playback(playback_matrix)
        self._store_playback(playback_matrix)
        reference_matrix = self._get_reference_playback()
        self._reference_chunk = reference_matrix

        processed_chunk = self.cancel_feedback(ADC, reference_matrix)
        packed_chunk = self.pack(self.chunk_number, processed_chunk)
        self.send(packed_chunk)

        self.play_chunk(DAC, playback_chunk)
        self._last_sent_chunk = processed_chunk
        self._maybe_store_calibration_recording(ADC)

    def _update_slider_value(self, slider, value):
        if slider is None:
            return
        self._slider_internal_update = True
        try:
            slider.set_val(value)
        finally:
            self._slider_internal_update = False

    def _start_slider_ui(self):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider
        except Exception as exc:  # pragma: no cover - optional GUI dependency
            logging.warning("Unable to start slider UI: %s", exc)
            return

        max_delay = max(0.0, float(minimal.args.feedback_max_delay_ms))
        if max_delay == 0.0:
            logging.info("feedback_max_delay_ms is 0. Slider UI disabled.")
            return
        fig, ax = plt.subplots(figsize=(7, 3))

        fig.suptitle("Feedback cancellation controls", fontsize=12)
        plt.subplots_adjust(left=0.14, bottom=0.25, top=0.83)
        ax.axis("off")

        att_ax = fig.add_axes([0.14, 0.15, 0.78, 0.05])
        delay_ax = fig.add_axes([0.14, 0.05, 0.78, 0.05])

        att_step = self._attenuation_step if self._attenuation_step > 0 else None
        delay_step = self._delay_step_ms if self._delay_step_ms > 0 else None

        self._attenuation_slider = Slider(
            att_ax,
            "Attenuation",
            self._attenuation_min,
            self._attenuation_max,
            valinit=self.attenuation,
            valstep=att_step,
        )
        self._delay_slider = Slider(
            delay_ax,
            "Delay (ms)",
            0.0,
            max_delay,
            valinit=self.delay_ms,
            valstep=delay_step,
        )

        fig.text(
            0.5,
            0.88,
            "Use the sliders for quick tuning; keyboard controls remain available.",
            ha="center",
            va="center",
            fontsize=9,
        )

        def _on_att_change(value):
            if self._slider_internal_update:
                return
            self.set_attenuation(value)

        def _on_delay_change(value):
            if self._slider_internal_update:
                return
            self.set_delay_ms(value)

        self._attenuation_slider.on_changed(_on_att_change)
        self._delay_slider.on_changed(_on_delay_change)

        def _on_close(event):
            logging.info("Slider window closed.")
            self._attenuation_slider = None
            self._delay_slider = None
            self._slider_fig = None

        fig.canvas.mpl_connect("close_event", _on_close)
        self._slider_fig = fig
        plt.show(block=False)
        logging.info("Matplotlib slider UI ready: attenuation and delay can be tuned visually.")

    def _pump_slider_events(self):
        if self._slider_fig is None:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        try:
            plt.pause(0.001)
        except Exception as exc:
            logging.debug("Slider UI event loop stopped: %s", exc)
            self._slider_fig = None
            self._attenuation_slider = None
            self._delay_slider = None

    def receive_and_buffer(self):
        chunk_number = super().receive_and_buffer()
        self._pump_slider_events()
        return chunk_number


class Feedback_Cancellation__verbose(Feedback_Cancellation, buffer.Buffering__verbose):
    def __init__(self):
        super().__init__()

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        if minimal.args.show_samples:
            self.show_recorded_chunk(ADC)

        super()._record_IO_and_play(ADC, DAC, frames, time, status)

        if minimal.args.show_samples:
            self.show_played_chunk(DAC)

        self.recorded_chunk = self._last_sent_chunk
        self.played_chunk = DAC
        self.reference_chunk = self._reference_chunk


try:
    import argcomplete
except ImportError:
    logging.warning("Unable to import argcomplete (optional)")


if __name__ == "__main__":
    minimal.parser.description = __doc__
    try:
        argcomplete.autocomplete(minimal.parser)
    except Exception:
        logging.warning("argcomplete not working :-/")
    minimal.args = minimal.parser.parse_known_args()[0]

    if minimal.args.show_stats or minimal.args.show_samples or minimal.args.show_spectrum:
        intercom = Feedback_Cancellation__verbose()
    else:
        intercom = Feedback_Cancellation()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()
