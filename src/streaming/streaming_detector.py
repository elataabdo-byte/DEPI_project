import numpy as np
import pandas as pd

import detect_scenario as ds
from rolling_buffer import RollingBuffer, history_length


class StreamingDetector:
    def __init__(self, model, window_size=None, window_seconds=None, stride=None, first_id=1, extra=8):
        self.model = model
        self.sensors = list(model["sensors"])        # the readings must be in this order (or a dict by name)
        self.buffer = RollingBuffer(self.sensors, window_size=window_size, stride=stride, first_id=first_id,
                                    window_seconds=window_seconds, history_windows=history_length(model, extra))

    def _values(self, reading):
        if isinstance(reading, dict):                 # digital / unused sensors in the dict are ignored
            return [reading[s] for s in self.sensors]
        return reading

    def push(self, reading, segment=0, timestamp=None):
        return [self._detect(w) for w in self.buffer.push(self._values(reading), segment, timestamp)]

    def flush(self):
        w = self.buffer.flush()
        return [] if w is None else [self._detect(w)]

    def _detect(self, win):
        m, s = win.history()                          # (windows x sensors), oldest first, last row = this window
        means = pd.DataFrame(m, columns=self.sensors)
        stds = pd.DataFrame(s, columns=self.sensors)
        analysis = ds.analyse_sensors(self.model, means, stds, np.zeros(len(means), dtype=int))
        res = ds.window_result(win.window_id, len(means) - 1, analysis, self.model)
        res.update(segment=win.segment, n_readings=win.n, start_ts=win.start_ts, end_ts=win.end_ts)
        return res
