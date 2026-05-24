"""Lightweight 1D Kalman filters for smoothing detection scores."""


class ScalarKalmanFilter:
    """Single-variable Kalman filter for scores in [0, 1]."""

    def __init__(
        self,
        initial: float = 0.0,
        process_noise: float = 0.02,
        measurement_noise: float = 0.15,
    ) -> None:
        self.x = initial
        self.p = 1.0
        self.q = process_noise
        self.r = measurement_noise

    def update(self, measurement: float) -> float:
        measurement = max(0.0, min(1.0, measurement))
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= 1.0 - k
        self.x = max(0.0, min(1.0, self.x))
        return self.x

    def reset(self, value: float = 0.0) -> None:
        self.x = max(0.0, min(1.0, value))
        self.p = 1.0
