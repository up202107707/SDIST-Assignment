import time
import threading
import subprocess
import re
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt

@dataclass
class ClockConfig:
    name: str
    color: str
    initial_offset: float = 15.0
    drift_ppm: float = 300.0
    kp: float = 0.20
    ki: float = 0.015
    sync_interval: float = 1.0


# ================================
# PI Calibration Function (independent & reusable)
# ================================
def tune_pi_controller(offset_history, dt=1.0, target_settling_time=30, max_overshoot_pct=10):
    """
    Very simple automatic PI tuner based on Ziegler-Nichols-like ideas.
    Use this separately to find good Kp/Ki before running long experiments.
    """
    if len(offset_history) < 10:
        return 0.15, 0.01

    # Estimate ultimate gain by increasing Kp until oscillation
    errors = [abs(e) for e in offset_history[-20:]]
    avg_error = sum(errors) / len(errors)

    Kp = 0.4
    Ki = 0.0
    if target_settling_time < 20:
        Kp = 0.25
        Ki = 0.018
    elif avg_error > 1e-4:  # large error → more aggressive
        Kp = 0.30
        Ki = 0.020

    print(f"[Auto-tune] Suggested Kp={Kp:.3f}, Ki={Ki:.3f} for settling ~{target_settling_time}s")
    return Kp, Ki


# ================================
# ServoClock class (cleaned & improved)
# ================================
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


    def _get_offset(self) -> Optional[float]:
        try:
            if "PTP" in self.config.name:
                out = subprocess.check_output(["pmc", "-u", "-b", "0", "GET TIME_STATUS_NP"], timeout=5).decode()
                m = re.search(r'master_offset\s+:\s+(-?\d+)', out)
                return abs(int(m.group(1)) * 1e-9) if m else None
            else:  # NTP
                out = subprocess.check_output(["chronyc", "tracking"], timeout=5).decode()
                m = re.search(r'Last offset\s+:\s+([+-]?\d*\.?\d+)', out)
                return abs(float(m.group(1))) if m else None
        except:
            return None

    def _apply_pi_correction(self, offset: float):
        self.integral += offset
        control = self.config.kp * offset + self.config.ki * self.integral
        self.rate_ppm = self.config.drift_ppm - control * 1e6
        self.virtual_time += self.config.kp * offset  # step + slew

    def stop(self):
        self.running = False
        self.thread.join()


# ================================
# Configuration
# ================================
ptp_cfg = ClockConfig("PTP Servo Clock", "tab:blue", initial_offset=15.0, drift_ppm=300.0,
                      kp=0.22, ki=0.016, sync_interval=1.0)
ntp_cfg = ClockConfig("NTP Servo Clock", "tab:red", initial_offset=15.0, drift_ppm=300.0,
                      kp=0.22, ki=0.016, sync_interval=8.0)

print("Starting PTP vs NTP Servo-Clock Comparison (Report-Ready Plots)")
print(f"Both clocks start with +{ptp_cfg.initial_offset}s offset and +{ptp_cfg.drift_ppm} ppm drift\n")

ptp = ServoClock(ptp_cfg)
ntp = ServoClock(ntp_cfg)

try:
    while True:
        time.sleep(10)
        print(f"PTP: {ptp.log[-1]['error_us']:6.1f} us   |   "
              f"NTP: {ntp.log[-1]['error_us']:7.1f} us", end="\r")
except KeyboardInterrupt:
    print("\n\nStopping clocks...")
    ptp.stop()
    ntp.stop()


# ================================
# REPORT-QUALITY PLOTS
# ================================
fig = plt.figure(figsize=(12, 8))

# --- Top: Real time vs Virtual clock time ---
ax1 = fig.add_subplot(2, 1, 1)
t0 = ptp.log[0]['mono_time']

ax1.plot([e['mono_time'] - t0 for e in ptp.log], [e['virtual_time'] - t0 for e in ptp.log],
         color=ptp_cfg.color, linewidth=2, label=f"{ptp_cfg.name}")
ax1.plot([e['mono_time'] - t0 for e in ntp.log], [e['virtual_time'] - t0 for e in ntp.log],
         color=ntp_cfg.color, linewidth=2, label=f"{ntp_cfg.name}")

# Reference line (perfect clock)
t_ref = [e['mono_time'] - t0 for e in ptp.log]
ax1.plot(t_ref, t_ref, 'k--', alpha=0.6, linewidth=1, label="Perfect clock")

ax1.set_title("Virtual Clock Time vs Real Monotonic Time\n"
              f"Initial offset = +{ptp_cfg.initial_offset}s, drift = +{ptp_cfg.drift_ppm} ppm")
ax1.set_ylabel("Virtual time (s)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# --- Bottom: Error evolution (us) ---
ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

err_ptp = [e['error_us'] for e in ptp.log]
err_ntp = [e['error_us'] for e in ntp.log]
t_ptp = [e['mono_time'] - t0 for e in ptp.log]
t_ntp = [e['mono_time'] - t0 for e in ntp.log]

ax2.plot(t_ptp, err_ptp, color=ptp_cfg.color, linewidth=2, label=f"PTP (sync every {ptp_cfg.sync_interval}s)")
ax2.plot(t_ntp, err_ntp, color=ntp_cfg.color, linewidth=2, label=f"NTP (sync every {ntp_cfg.sync_interval}s)")

ax2.set_title("Clock Synchronization Error Evolution")
ax2.set_xlabel("Experiment time (seconds)")
ax2.set_ylabel("Error (us)")
ax2.grid(True, alpha=0.3)
ax2.legend()

# Stats
plt.figtext(0.62, 0.02,
            f"PTP final: {err_ptp[-1]:.1f} us (max {max(abs(e) for e in err_ptp):.1f} us)  |  "
            f"NTP final: {err_ntp[-1]:.1f} us (max {max(abs(e) for e in err_ntp):.1f} us)",
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle("PTP vs NTP User-Level Servo-Clock Synchronization", fontsize=16, y=0.98)
plt.show()

# Optional: run auto-tuner on collected data
print("\nRunning PI auto-calibration suggestion on last 30 seconds...")
recent_offsets = [e['error_us']*1e-6 for e in ptp.log[-300:]]
tune_pi_controller(recent_offsets, target_settling_time=25)