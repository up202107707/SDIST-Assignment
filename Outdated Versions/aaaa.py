import threading
import time
import random
import numpy as np
import matplotlib.pyplot as plt


class ServoClock:
    def __init__(self, initial_offset=0.0, drift_rate=5e-6):
        self.time_seconds = initial_offset
        self.drift_rate = drift_rate
        self.freq_correction = 0.0
        self.lock = threading.Lock()
        self.noise_std = 1e-6

    def tick(self, dt):
        with self.lock:
            actual_drift = self.drift_rate + self.freq_correction
            self.time_seconds += dt * (1.0 + actual_drift) + random.gauss(0, self.noise_std)

    def get_time(self):
        with self.lock:
            return self.time_seconds

    def apply_correction(self, offset, kp=0.1, ki=0.01, dt=1.0):
        with self.lock:
            self.freq_correction += ki * offset * dt
            self.time_seconds += kp * offset



class ClockMetrics:
    def __init__(self, name_i="Clock i", name_j="Clock j"):
        self.name_i = name_i
        self.name_j = name_j
        self.times_i = [] 
        self.times_j = [] 
        self.offsets = [] 

    def sample(self, time_i, time_j):
        self.times_i.append(time_i)
        self.times_j.append(time_j)
        self.offsets.append(abs(time_i - time_j))

    def get_offset(self):
        if not self.offsets:
            return None
        return self.offsets[-1]

    def get_drift(self):
        if len(self.offsets) < 2:
            return None
        return abs(self.offsets[-1] - self.offsets[0])

    def get_drift_rate(self):
        if len(self.offsets) < 2:
            return None
        dt = self.times_i[-1] - self.times_i[0]
        if dt == 0:
            return None
        return abs(self.offsets[-1] - self.offsets[0]) / dt

    def get_accuracy(self):
        if not self.offsets:
            return None
        return np.max(self.offsets)

    def get_precision(self):
        if not self.offsets:
            return None
        return np.std(self.offsets)

    def summary(self):
        print(f"--- Metrics between {self.name_i} and {self.name_j} ---")
        print(f"Current Offset: {1e3*self.get_offset():.6f} ms")
        print(f"Drift: {1e3*self.get_drift():.6f} ms")
        print(f"Drift Rate: {1e6*self.get_drift_rate():.6f} ppm")
        print(f"Accuracy (max offset): {1e3*self.get_accuracy():.6f} ms")
        print(f"Precision (std dev): {1e3*self.get_precision():.6f} ms")



# --- ServoClock and ClockMetrics already defined ---

# Create a servo clock and metrics tracker
servo = ServoClock(drift_rate=5e-3) 
metrics = ClockMetrics("Servo", "Global")

servo_time = 0.0
global_time = 0.0
steps = 50
dt = 0.01

# Simulation loop
for step in range(steps):
    global_time += dt
    servo.tick(dt)

    # Compute offset (you can uncomment correction if needed)
    offset = global_time - servo.get_time()
    # servo.apply_correction(offset, kp=0.2, ki=0.05, dt=dt)

    # Sample metrics
    metrics.sample(servo.get_time(), global_time)

# Print final metrics (all absolute values)
metrics.summary()

