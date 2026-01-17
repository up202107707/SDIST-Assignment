import subprocess
import socket
import struct
import time
import re
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Time source interface (abstract base class)
class TimeSource(ABC):
    def __init__(self, noise_std=0.0):
        self.noise_std = noise_std

    @abstractmethod
    def get_time(self):
        pass

    @abstractmethod
    def name(self):
        pass

    def _add_noise(self, value):
        if self.noise_std > 0:
            return value + random.gauss(0, self.noise_std)
        return value


# NTP client implementation
class NTPSource(TimeSource):
    NTP_DELTA = 2208988800
    NTP_QUERY = b'\x1b' + 47 * b'\0'

    def __init__(self, server="pool.ntp.org", noise_std=0.0):
        super().__init__(noise_std)
        self.server = server

    def get_time(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(2)
                t1 = time.time()
                s.sendto(self.NTP_QUERY, (self.server, 123))
                msg, _ = s.recvfrom(1024)
                t4 = time.time()

            if len(msg) < 48:
                raise ValueError("Invalid NTP response")

            unpacked = struct.unpack("!12I", msg[:48])
            t3 = unpacked[10] + (unpacked[11] / 2**32) - self.NTP_DELTA
            t2 = unpacked[8] + (unpacked[9] / 2**32) - self.NTP_DELTA

            offset = ((t2 - t1) + (t3 - t4)) / 2
            offset = self._add_noise(offset)
            return time.time() + offset

        except Exception as e:
            print(f"[NTP] Error querying {self.server}: {e}")
            return time.time()

    def name(self):
        return "NTP"


# PTP time source (uses pmc command)
class PTPSource(TimeSource):
    OFFSET_REGEXES = [
        re.compile(r"master_offset\s+([-+]?\d+)"),
        re.compile(r"master offset\s+([-+]?\d+)"),
        re.compile(r"offset\s+([-+]?\d+)")
    ]

    def __init__(self, interface="eth0", noise_std=0.0):
        super().__init__(noise_std)
        self.interface = interface

    def get_time(self):
        try:
            cmd = ["sudo", "pmc", "-u", "-b", "0", "GET TIME_STATUS_NP"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)

            offset_ns = None
            for regex in self.OFFSET_REGEXES:
                match = regex.search(result.stdout)
                if match:
                    offset_ns = int(match.group(1))
                    break

            if offset_ns is None:
                raise ValueError("Could not parse PTP offset")

            offset_s = self._add_noise(offset_ns * 1e-9)
            return time.time() + offset_s

        except Exception as e:
            print(f"[PTP] Error reading time: {e}")
            return time.time()

    def name(self):
        return "PTP"


# Simple PI controller for frequency adjustment
@dataclass
class PIController:
    Kp: float
    Ki: float
    integral: float = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        return self.Kp * error + self.Ki * self.integral


# Local clock with frequency correction
class ServoClock:
    def __init__(self, initial_time, drift_ppm, controller):
        self.drift = drift_ppm * 1e-6
        self.controller = controller
        self.time = initial_time
        self.rate_correction = 0.0

    def update(self, dt, target_time=None):
        if target_time is not None:
            error = target_time - self.time
            self.rate_correction = self.controller.update(error, dt)

        self.time += dt * (1.0 + self.drift + self.rate_correction)

    def current_time(self):
        return self.time


# Uncorrected reference clock (for comparison)
class BaselineClock:
    def __init__(self, initial_time, drift_ppm):
        self.drift = drift_ppm * 1e-6
        self.time = initial_time

    def update(self, dt):
        self.time += dt * (1.0 + self.drift)

    def current_time(self):
        return self.time


# Synchronization performance metrics
@dataclass
class SyncMetrics:
    mean_abs_offset: float
    std_dev: float
    max_abs_offset: float
    min_abs_offset: float
    convergence_time: Optional[float]
    steady_state_offset: float
    jitter: float
    settling_time_1pct: Optional[float]
    settling_time_01pct: Optional[float]

    @staticmethod
    def compute(offsets, sync_times, sync_exp_times, initial_offset,
                abs_threshold=0.001, relative_thresholds=[1.0, 0.1]):
        
        if not offsets:
            return SyncMetrics(0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, None, None)

        arr = np.array(offsets)
        abs_arr = np.abs(arr)

        mean_abs = np.mean(abs_arr)
        std = np.std(arr)
        max_abs = np.max(abs_arr)
        min_abs = np.min(abs_arr)

        # Time when offset stays under absolute threshold forever
        conv_time = None
        for i in range(len(offsets)):
            if all(abs(o) < abs_threshold for o in offsets[i:]):
                conv_time = sync_exp_times[i]
                break

        # Steady-state (last 30%)
        steady_idx = int(len(offsets) * 0.7)
        ss_offset = np.mean(abs_arr[steady_idx:])
        jitter = np.std(arr[steady_idx:])

        # Relative settling times
        initial_abs = abs(initial_offset) if abs(initial_offset) > 1e-9 else 1.0
        settling = {}

        for pct in relative_thresholds:
            band = (pct / 100.0) * initial_abs
            t = None
            for i in range(len(offsets)):
                if all(abs(o) <= band for o in offsets[i:]):
                    t = sync_exp_times[i]
                    break
            settling[pct] = t

        return SyncMetrics(
            mean_abs, std, max_abs, min_abs,
            conv_time, ss_offset, jitter,
            settling.get(1.0), settling.get(0.1)
        )

    def print_summary(self, title, initial_offset):
        init_abs = abs(initial_offset)
        print(f"\n{'=' * 70}")
        print(f"  {title} - Synchronization Statistics")
        print(f"  Initial offset: {init_abs:.1f} seconds")
        print(f"{'=' * 70}")
        print(f"  Mean |offset|:              {self.mean_abs_offset*1000:8.3f} ms")
        print(f"  Standard deviation:         {self.std_dev*1000:8.3f} ms")
        print(f"  Maximum |offset|:           {self.max_abs_offset*1000:8.3f} ms")
        print(f"  Steady-state |offset|:      {self.steady_state_offset*1000:8.3f} ms")
        print(f"  Steady-state jitter:        {self.jitter*1000:8.3f} ms")

        print(f"  Convergence (<1ms):         ", end="")
        print(f"{self.convergence_time:6.2f} s" if self.convergence_time else "Not achieved")

        print(f"  Settling ±1%:               ", end="")
        print(f"{self.settling_time_1pct:6.2f} s" if self.settling_time_1pct else "Not achieved")

        print(f"  Settling ±0.1%:             ", end="")
        print(f"{self.settling_time_01pct:6.2f} s" if self.settling_time_01pct else "Not achieved")
        print(f"{'=' * 70}\n")


# Experiment execution logic
class ExperimentRunner:
    def __init__(self, time_source, servo, sampling_period, sync_period,
                 duration, start_delay, baseline, experiment_name, initial_offset):
        self.time_source = time_source
        self.servo = servo
        self.baseline = baseline
        self.sampling_period = sampling_period
        self.sync_period = sync_period
        self.duration = duration
        self.start_delay = start_delay
        self.experiment_name = experiment_name
        self.initial_offset = initial_offset

        self.exp_times = []
        self.servo_times = []
        self.baseline_times = []
        self.sync_times = []
        self.sync_exp_times = []
        self.offsets = []

    def run(self):
        print(f"\nStarting: {self.experiment_name}")
        print(f"Duration: {self.duration}s  Sampling: {self.sampling_period}s  Sync: {self.sync_period}s")
        print(f"Initial offset: {self.initial_offset:.1f} s\n")

        start_time = time.time()
        next_sample = start_time
        next_sync = start_time + self.start_delay
        prev_time = start_time

        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed > self.duration:
                break

            dt = now - prev_time
            prev_time = now

            target_time = None
            if elapsed >= self.start_delay and now >= next_sync:
                target_time = self.time_source.get_time()
                offset = target_time - self.servo.time
                self.offsets.append(offset)
                self.sync_times.append(target_time)
                self.sync_exp_times.append(elapsed)
                next_sync += self.sync_period
                print(f"[t={elapsed:.3f}s]  SYNC  offset: {offset:.6f}s")

            self.servo.update(dt, target_time)
            self.baseline.update(dt)

            self.exp_times.append(elapsed)
            self.servo_times.append(self.servo.time)
            self.baseline_times.append(self.baseline.time)

            next_sample += self.sampling_period
            sleep_time = next_sample - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("\nExperiment finished.")

        if self.offsets:
            metrics = SyncMetrics.compute(
                self.offsets, self.sync_times, self.sync_exp_times, self.initial_offset
            )
            metrics.print_summary(f"{self.time_source.name()} - {self.experiment_name}", self.initial_offset)

            # Save data to CSV
            df = pd.DataFrame({
                'sync_exp_time': self.sync_exp_times,
                'offset': self.offsets
            })
            csv_filename = f"{self.time_source.name()}_{self.experiment_name.replace(' ', '_').replace(':', '')}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")

            return metrics, self.sync_exp_times, self.offsets
        else:
            print("No sync data collected.")
            return None, None, None


# Plot offset comparison between protocols for a specific variation
def plot_comparison(ntp_data, ptp_data, variation_name):
    if not ntp_data or not ptp_data:
        return

    ntp_t, ntp_off = ntp_data
    ptp_t, ptp_off = ptp_data

    plt.figure(figsize=(10, 6))
    plt.plot(ntp_t, [o*1000 for o in ntp_off], label="NTP", color="blue", marker='o')
    plt.plot(ptp_t, [o*1000 for o in ptp_off], label="PTP", color="red", marker='x')
    plt.xlabel("Experiment time (s)")
    plt.ylabel("Offset (ms)")
    plt.title(f"Offset Comparison - {variation_name}")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

# Plot all variations for a single protocol
def plot_all_variations(protocol_name, plot_data_dict):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors  # Use a colormap for different variations

    for i, (var_name, (times, offsets)) in enumerate(plot_data_dict.items()):
        plt.plot(times, [o*1000 for o in offsets], label=var_name, color=colors[i % len(colors)], marker='o', linestyle='-')

    plt.xlabel("Experiment time (s)")
    plt.ylabel("Offset (ms)")
    plt.title(f"All Variations - {protocol_name} Offsets")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

# Print comparison table for multiple parameter sets
def print_metrics_table(protocol_name, results):
    print(f"\n{'='*100}")
    print(f"  {protocol_name} - Results for different parameter sets")
    print(f"{'='*100}")
    headers = ["Variation", "Mean |offset|", "Std", "Max |offset|",
               "Steady-state", "Jitter", "Conv 1ms", "Settle ±1%", "Settle ±0.1%"]

    rows = []
    for name, m in results:
        conv = f"{m.convergence_time:.2f}" if m.convergence_time else "N/A"
        s1 = f"{m.settling_time_1pct:.2f}" if m.settling_time_1pct else "N/A"
        s01 = f"{m.settling_time_01pct:.2f}" if m.settling_time_01pct else "N/A"

        rows.append([
            name,
            f"{m.mean_abs_offset*1000:.3f}",
            f"{m.std_dev*1000:.3f}",
            f"{m.max_abs_offset*1000:.3f}",
            f"{m.steady_state_offset*1000:.3f}",
            f"{m.jitter*1000:.3f}",
            conv, s1, s01
        ])

    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    print(" | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)))
    print("-" * (sum(col_widths) + len(headers)*3 - 1))
    for row in rows:
        print(" | ".join(f"{v:<{w}}" for v, w in zip(row, col_widths)))
    print(f"{'='*100}\n")

# Run multiple experiments with different parameters
def run_experiments(time_source, base_config, param_sets):
    results = []
    plot_data = {}

    for i, params in enumerate(param_sets, 1):
        config = {**base_config, **params}
        name = f"Var {i}: Kp={config['Kp']}, Ki={config['Ki']}, Sync={config['sync_period']}s"

        ref_time = time_source.get_time()
        init_offset = config["initial_offset"]
        init_servo_time = ref_time + init_offset

        controller = PIController(config["Kp"], config["Ki"])
        servo = ServoClock(init_servo_time, config["drift_ppm"], controller)
        baseline = BaselineClock(init_servo_time, config["drift_ppm"])

        runner = ExperimentRunner(
            time_source, servo, config["sampling_period"], config["sync_period"],
            config["duration"], config["start_delay"], baseline, name, init_offset
        )

        metrics, t_sync, offsets = runner.run()
        if metrics:
            results.append((name, metrics))
            plot_data[name] = (t_sync, offsets)

    return results, plot_data


def main():
    config = {
        "sampling_period": 0.1,
        "duration": 60.0,
        "drift_ppm": 300.0,
        "initial_offset": 20.0,
        "start_delay": 5.0,
        "ntp_server": "127.0.0.1",
        "ptp_interface": "eth0",
        "offset_noise_std": 100e-6
    }

    variations = [
        {"Kp": 0.1,  "Ki": 0.05,  "sync_period": 5.0},
        {"Kp": 0.2,  "Ki": 0.1,   "sync_period": 5.0},
        {"Kp": 0.05, "Ki": 0.025, "sync_period": 5.0},
        {"Kp": 0.1,  "Ki": 0.05,  "sync_period": 2.0},
        {"Kp": 0.1,  "Ki": 0.05,  "sync_period": 10.0},
    ]

    print("=" * 70)
    print("          NTP and PTP Servo Clock Synchronization Experiments")
    print("=" * 70)
    print()

    # Run NTP experiments
    ntp_source = NTPSource(config["ntp_server"], config["offset_noise_std"])
    ntp_results, ntp_plot_data = run_experiments(ntp_source, config, variations)
    print_metrics_table("NTP", ntp_results)

    # Show all variations for NTP
    plot_all_variations("NTP", ntp_plot_data)

    # Run PTP experiments
    ptp_source = PTPSource(config["ptp_interface"], config["offset_noise_std"])
    ptp_results, ptp_plot_data = run_experiments(ptp_source, config, variations)
    print_metrics_table("PTP", ptp_results)

    # Show all variations for PTP
    plot_all_variations("PTP", ptp_plot_data)

    # Show per-variation comparison plots (NTP vs PTP)
    for i in range(len(variations)):
        name = f"Var {i+1}"
        plot_comparison(ntp_plot_data.get(name), ptp_plot_data.get(name), name)

    print("\nAll experiments finished. Close plot windows to exit.\n")
    plt.show()


if __name__ == "__main__":
    main()