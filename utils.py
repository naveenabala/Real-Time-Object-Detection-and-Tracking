%%writefile utils.py
import time
from collections import deque


class FPSMeter:
    def __init__(self, window=30):
        self.timestamps = deque(maxlen=window)

    def tick(self):
        self.timestamps.append(time.time())

    @property
    def fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def _side_of_line(point, line_start, line_end):
    (x1, y1), (x2, y2) = line_start, line_end
    x, y = point
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


class LineCounter:
    def __init__(self, line_start, line_end):
        self.line_start = line_start
        self.line_end = line_end
        self._last_side = {}
        self.count_a_to_b = 0
        self.count_b_to_a = 0

    def update(self, track_id, centroid):
        side = _side_of_line(centroid, self.line_start, self.line_end)
        side_sign = 1 if side > 0 else -1

        prev_sign = self._last_side.get(track_id)
        if prev_sign is not None and prev_sign != side_sign:
            if prev_sign == -1 and side_sign == 1:
                self.count_a_to_b += 1
            else:
                self.count_b_to_a += 1

        self._last_side[track_id] = side_sign

    @property
    def total(self):
        return self.count_a_to_b + self.count_b_to_a
