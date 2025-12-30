import time
import threading
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
import random

@dataclass
class ClockConfig:
    name: str
    color: str
    initial_offset: float = 15.0
    drift_ppm: float = 300.0
    kp: float = 0.20
    ki: float = 0.015
    sync_interval: float = 1.0
    simulated: bool = True   # If True, generates offsets instead of calling system commands

# --------------------------
# PI Auto-tuner (same)
# --------------------------
def tune_pi_controller(offset_history, dt=1.0, target_settling_time=30, max_overshoot_pct=10):
    if len(offset_history) < 10:
        return 0.15, 0.01
    errors = [abs(e) for e in offset_history[-20:]]
    avg_error = sum(errors) / len(errors)
    Kp, Ki = 0.3, 0.02 if avg_error > 1e-4 else (0.25, 0.015)
    print(f"[Auto-tune] Suggested Kp={Kp:.3f}, Ki={Ki:.3f}")
    return Kp, Ki

# --------------------------
# ServoClock Class
# --------------------------
class ServoClock:
    def __init__(self, config: ClockConfig):
        self.config = config
        self.virtual_time = time.monotonic() + config.initial_offset
        self.rate_ppm = config.drift_ppm
        self.integral = 0.0
        self.last_sync = time.monotonic()
        self.log = []
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            offset = None
            now = time.monotonic()
            # --------------------
            # Sync
            # --------------------
            if now - self.last_sync >= self.config.sync_interval:
                offset = self._get_offset()
                if offset is not None:
                    self._apply_pi_correction(offset)
                self.last_sync = now

            dt = time.monotonic() - now
            self.virtual_time += dt * (1.0 + self.rate_ppm * 1e-6)

            error = self.virtual_time - time.monotonic()
            self.log.append({
                'wall_time': time.time(),
                'mono_time': time.monotonic(),
                'virtual_time': self.virtual_time,
                'error_us': error * 1e6,
                'offset_us': offset * 1e6 if offset is not None else None,
                'rate_ppm': self.rate_ppm
            })
            time.sleep(0.05)

    # --------------------------
    # Offset provider
    # --------------------------
    def _get_offset(self) -> Optional[float]:
        if self.config.simulated:
            # simulate small offset fluctuations
            return random.gauss(0, 0.00005)  # +/-50 µs
        else:
            # placeholder for real PTP/NTP
            return 0.0

    # --------------------------
    # PI correction
    # --------------------------
    def _apply_pi_correction(self, offset: float):
        self.integral += offset
        control = self.config.kp * offset + self.config.ki * self.integral
        self.rate_ppm = self.config.drift_ppm - control * 1e6
        self.virtual_time += self.config.kp * offset

    def stop(self):
        self.running = False
        self.thread.join()


# --------------------------
# Simulation Config
# --------------------------
ptp_cfg = ClockConfig("PTP Servo Clock", "tab:blue", sync_interval=1.0, simulated=True)
ntp_cfg = ClockConfig("NTP Servo Clock", "tab:red", sync_interval=8.0, simulated=True)

ptp = ServoClock(ptp_cfg)
ntp = ServoClock(ntp_cfg)

# Run for a fixed simulation time (e.g., 60s)
sim_time = 10
start_time = time.time()
try:
    while time.time() - start_time < sim_time:
        time.sleep(2)
        if ptp.log and ntp.log:
            print(f"PTP: {ptp.log[-1]['error_us']:6.1f} us   |   NTP: {ntp.log[-1]['error_us']:6.1f} us", end="\r")
except KeyboardInterrupt:
    pass

ptp.stop()
ntp.stop()

# --------------------------
# Plotting
# --------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

t0 = ptp.log[0]['mono_time']

ax1.plot([e['mono_time'] - t0 for e in ptp.log], [e['virtual_time'] - t0 for e in ptp.log],
         color=ptp_cfg.color, linewidth=2, label=ptp_cfg.name)
ax1.plot([e['mono_time'] - t0 for e in ntp.log], [e['virtual_time'] - t0 for e in ntp.log],
         color=ntp_cfg.color, linewidth=2, label=ntp_cfg.name)
ax1.plot([e['mono_time'] - t0 for e in ptp.log],
         [e['mono_time'] - t0 for e in ptp.log],
         'k--', alpha=0.6, label="Perfect clock")
ax1.set_ylabel("Virtual time (s)")
ax1.set_title("Virtual Clock vs Monotonic Time")
ax1.grid(True)
ax1.legend()

err_ptp = [e['error_us'] for e in ptp.log]
err_ntp = [e['error_us'] for e in ntp.log]
t = [e['mono_time'] - t0 for e in ptp.log]

ax2.plot(t, err_ptp, color=ptp_cfg.color, linewidth=2, label=f"PTP (sync {ptp_cfg.sync_interval}s)")
ax2.plot(t, err_ntp, color=ntp_cfg.color, linewidth=2, label=f"NTP (sync {ntp_cfg.sync_interval}s)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Error (us)")
ax2.set_title("Clock Synchronization Error")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# --------------------------
# Auto-tune
# --------------------------
recent_offsets = [e['error_us']*1e-6 for e in ptp.log[-300:]]
tune_pi_controller(recent_offsets)
