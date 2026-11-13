import warnings
from collections import deque

import numpy as np


def history_length(model, extra=8):
    """How many past windows the scenario rules need (same formula the old streaming code used)."""
    p = model["params"]
    flat = [b["fault_flat"] for b in model["sensors"].values() if b.get("flat_rule_enabled")]
    longest = max([p["fault_min_windows"], p["L"] + p["shift_m"], p["drift_N"], p["rep_horizon"]] + flat)
    return longest + p["L"] + extra


def to_seconds(timestamp):
    """A number (seconds), a datetime string like '2023-11-26 10:57:52', a datetime64 or a pandas Timestamp -> float seconds."""
    try:
        return float(timestamp)
    except (TypeError, ValueError):
        return float(np.datetime64(timestamp, "s").astype("int64"))


class Window:
    __slots__ = ("window_id", "segment", "n", "start_ts", "end_ts", "readings", "means", "stds", "_history")

    def __init__(self, window_id, segment, n, start_ts, end_ts, readings, means, stds, history):
        self.window_id, self.segment, self.n = window_id, segment, n
        self.readings = readings            # the raw readings of the window, shape (n, number of sensors)
        self.start_ts, self.end_ts = start_ts, end_ts
        self.means, self.stds = means, stds
        self._history = history              # snapshot taken when the window was built

    def history(self):
        """(means, stds) of the last windows of this segment, oldest first, ending with this window."""
        h = self._history
        return np.array([m for m, _ in h]), np.array([s for _, s in h])


class RollingBuffer:
    def __init__(self, sensors, window_size=None, stride=None, history_windows=0, first_id=1, window_seconds=None):
        self.sensors = list(sensors)
        self.seconds = window_seconds          # time mode: a window = every reading in the same N-second bin
        if self.seconds is None and window_size is None:
            raise ValueError("give window_size (count mode) or window_seconds (time mode)")
        self.size = int(window_size or 1)
        self.stride = int(stride or self.size)
        if self.stride > self.size:
            raise ValueError("stride bigger than window_size would skip readings")
        self._tm_rows, self._tm_times = [], []   # time mode buffers
        self._seg_start = None
        self._last_t = None
        self._bin = None
        self.past = deque(maxlen=max(history_windows, 1))   # (means, stds) of recent windows
        self._rows = np.empty((self.size, len(self.sensors)))
        self._times = [None] * self.size
        self._count = 0            # readings currently in the buffer
        self._since_emit = 0       # readings since the last emitted window (for stride < size)
        self.segment = None
        self.next_id = first_id

    def _summary(self, rows):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN column -> NaN, like pandas
            return np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)   # ddof=0, same as detect_scenario

    def _emit(self, rows, times):
        means, stds = self._summary(rows)
        self.past.append((means, stds))
        win = Window(self.next_id, self.segment, len(rows), times[0], times[-1], rows, means, stds, list(self.past))
        self.next_id += 1
        return win

    def push(self, values, segment=0, timestamp=None):
        """Add one reading. Returns a list with 0, 1 or 2 completed Windows (2 only when a new segment
        closes the old segment's partial window and the new reading also completes a window)."""
        if self.seconds is not None:
            return self._push_time(values, segment, timestamp)
        done = []
        if self.segment is not None and segment != self.segment:
            tail = self.flush()                 # close the partial window of the old segment
            if tail is not None:
                done.append(tail)
            self.past.clear()                   # history never crosses a segment
        self.segment = segment
        if self._count == self.size:            # overlapping windows: slide by dropping the oldest readings
            keep = self.size - self.stride
            self._rows[:keep] = self._rows[self.stride:]
            self._times[:keep] = self._times[self.stride:]
            self._count = keep
        self._rows[self._count] = values
        self._times[self._count] = timestamp
        self._count += 1
        self._since_emit += 1
        if self._count == self.size and self._since_emit >= self.stride:
            self._since_emit = 0
            done.append(self._emit(self._rows[:self._count].copy(), self._times[:self._count]))
            if self.stride == self.size:
                self._count = 0
        return done

    # ---- time mode: windows are fixed N-second bins counted from the first reading of the segment ----
    def _push_time(self, values, segment, timestamp):
        t = to_seconds(timestamp)               # seconds (epoch or any increasing clock)
        done = []
        if self.segment is not None and segment != self.segment:
            tail = self._flush_time()
            if tail is not None:
                done.append(tail)
            self.past.clear()                   # history never crosses a segment
            self._seg_start = None
        if self._seg_start is None:
            self._seg_start = t
            self._last_t = t
        if t < self._last_t:
            raise ValueError(f"readings must arrive in time order (got {t} after {self._last_t})")
        self._last_t = t
        self.segment = segment
        b = int((t - self._seg_start) // self.seconds)
        if self._bin is not None and b != self._bin:      # first reading of the next bin closes this window
            done.append(self._flush_time())
        self._bin = b
        self._tm_rows.append(np.asarray(values, dtype=float))
        self._tm_times.append(t)
        return done

    def _flush_time(self):
        if not self._tm_rows:
            return None
        win = self._emit(np.array(self._tm_rows), self._tm_times)
        self._tm_rows, self._tm_times, self._bin = [], [], None
        return win

    def flush(self):
        """Emit the readings still waiting (a partial window), e.g. at the end of a segment or the stream."""
        if self.seconds is not None:
            return self._flush_time()
        if self._count == 0 or self._since_emit == 0:
            self._count = self._since_emit = 0
            return None
        win = self._emit(self._rows[:self._count].copy(), self._times[:self._count])
        self._count = self._since_emit = 0
        return win
