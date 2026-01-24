import subprocess
import socket
import struct
import time
import re
import random
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================
DEFAULT_CONFIG = {
    "sampling_period": 0.1,
    "duration": 180.0,
    "drift_ppm": 300.0,
    "initial_offset": 20.0,
    "start_delay": 5.0,
    "offset_noise_std": 200e-6,
    "ntp_timeout": 5.0,
    "ptp_timeout": 5.0,
    "max_consecutive_failures": 3,
    "ntp_pool": [
        "0.pt.pool.ntp.org",
        "1.pt.pool.ntp.org",
        "2.pt.pool.ntp.org",
        "time.cloudflare.com",
        "pool.ntp.org"
    ],
    "ptp_interface": "enxb6d40bb65a22",
    "status_interval": 10.0,
}


# ============================================================
# Time Source Abstraction
# ============================================================
class TimeSource(ABC):
    """Base class for time synchronization sources with error handling"""

    def __init__(self, noise_std: float = 0.0, timeout: float = 5.0, max_failures: int = 3):
        self.noise_std = noise_std
        self.timeout = timeout
        self.max_failures = max_failures
        self.consecutive_failures = 0
        self.total_queries = 0
        self.successful_queries = 0
        self.last_good_time = None
        self.last_good_timestamp = None

    def query_time(self) -> float:
        """Query time with automatic fallback on failure"""
        self.total_queries += 1

        try:
            result = self._query_time_impl()
            if result is not None:
                self.consecutive_failures = 0
                self.successful_queries += 1
                self.last_good_time = result
                self.last_good_timestamp = time.time()
                return result
            else:
                self._handle_failure("Query returned None")
                return self._get_fallback_time()

        except Exception as e:
            self._handle_failure(str(e))
            return self._get_fallback_time()

    def _handle_failure(self, error_msg: str):
        """Handle query failure"""
        self.consecutive_failures += 1
        print(f"  [{self.name()}] Query failed ({self.consecutive_failures}/{self.max_failures}): {error_msg}")

        if self.consecutive_failures >= self.max_failures:
            print(f"  [{self.name()}] WARNING: Max failures reached, using fallback")

    def _get_fallback_time(self) -> float:
        """Return fallback time based on last good measurement"""
        if self.last_good_time and self.last_good_timestamp:
            elapsed = time.time() - self.last_good_timestamp
            return self.last_good_time + elapsed
        return time.time()

    def get_success_rate(self) -> float:
        """Calculate query success rate percentage"""
        if self.total_queries == 0:
            return 0.0
        return (self.successful_queries / self.total_queries) * 100

    def _add_noise(self, value: float) -> float:
        """Add Gaussian noise to simulate measurement uncertainty"""
        if self.noise_std > 0:
            return value + random.gauss(0, self.noise_std)
        return value

    @abstractmethod
    def _query_time_impl(self) -> Optional[float]:
        """Implementation-specific time query"""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return protocol name"""
        pass

class NTPSource(TimeSource):
    """NTP client with automatic server failover"""

    NTP_DELTA = 2208988800  # Seconds between 1900 and 1970
    NTP_QUERY = b'\x1b' + 47 * b'\0'

    def __init__(self, server_pool: List[str], noise_std: float = 0.0,
                 timeout: float = 5.0, max_failures: int = 3):
        super().__init__(noise_std, timeout, max_failures)
        self.server_pool = server_pool if isinstance(server_pool, list) else [server_pool]
        self.current_server_idx = 0
        self.server_failures = {s: 0 for s in self.server_pool}

        print(f"[NTP] Initialized with {len(self.server_pool)} servers")
        print(f"[NTP] Primary: {self.server_pool[0]}")

    def _get_current_server(self) -> str:
        """Get server with fewest failures"""
        min_failures = min(self.server_failures.values())
        best_servers = [s for s, f in self.server_failures.items() if f == min_failures]

        if len(best_servers) > 0:
            return best_servers[self.current_server_idx % len(best_servers)]
        return self.server_pool[0]

    def _rotate_server(self):
        """Rotate to next server in pool"""
        self.current_server_idx = (self.current_server_idx + 1) % len(self.server_pool)
        print(f"[NTP] Rotating to server: {self._get_current_server()}")

    def _query_time_impl(self) -> Optional[float]:
        """Query NTP server using SNTP protocol"""
        server = self._get_current_server()

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(self.timeout)

                t1 = time.time()
                sock.sendto(self.NTP_QUERY, (server, 123))

                msg, _ = sock.recvfrom(1024)
                t4 = time.time()

                if len(msg) < 48:
                    raise ValueError(f"Invalid NTP response length: {len(msg)}")

                unpacked = struct.unpack("!12I", msg[:48])

                t2 = unpacked[8] + (unpacked[9] / 2**32) - self.NTP_DELTA
                t3 = unpacked[10] + (unpacked[11] / 2**32) - self.NTP_DELTA

                offset = ((t2 - t1) + (t3 - t4)) / 2
                offset = self._add_noise(offset)

                result = time.time() + offset
                self.server_failures[server] = 0

                return result

        except socket.timeout:
            self.server_failures[server] += 1
            self._rotate_server()
            raise TimeoutError(f"NTP timeout: {server}")

        except socket.gaierror as e:
            self.server_failures[server] += 1
            self._rotate_server()
            raise ConnectionError(f"DNS failed for {server}: {e}")

        except Exception as e:
            self.server_failures[server] += 1
            raise RuntimeError(f"NTP error with {server}: {e}")

    def name(self) -> str:
        return "NTP"

class PTPSource(TimeSource):
    """PTP client using pmc (PTP Management Client)"""

    def __init__(self, interface: str, noise_std: float = 0.0,
                 timeout: float = 5.0, max_failures: int = 3):
        super().__init__(noise_std, timeout, max_failures)
        self.interface = interface
        self._check_ptp_availability()

    def _check_ptp_availability(self):
        """Verify pmc is available"""
        try:
            result = subprocess.run(["which", "pmc"], capture_output=True, timeout=2)
            if result.returncode != 0:
                print("[PTP] WARNING: 'pmc' not found. Install linuxptp.")
        except Exception as e:
            print(f"[PTP] WARNING: Cannot verify pmc: {e}")

    def _query_time_impl(self) -> Optional[float]:
        """Query PTP daemon via pmc"""
        try:
            cmd = ["sudo", "pmc", "-u", "-b", "0", "GET TIME_STATUS_NP"]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)

            if result.returncode != 0:
                raise RuntimeError(f"pmc failed: {result.stderr.strip()}")

            offset_ns = self._parse_offset(result.stdout)

            if offset_ns is None:
                raise ValueError("Could not parse PTP offset")

            offset_s = self._add_noise(offset_ns * 1e-9)
            return time.time() + offset_s

        except subprocess.TimeoutExpired:
            raise TimeoutError("PTP pmc timeout")

        except FileNotFoundError:
            raise RuntimeError("pmc not found - install linuxptp")

        except Exception as e:
            raise RuntimeError(f"PTP error: {e}")

    def _parse_offset(self, output: str) -> Optional[int]:
        """Parse master_offset from pmc output"""
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("master_offset"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[-1])
        return None

    def name(self) -> str:
        return "PTP"


# ============================================================
# Clock Control and Synchronization
# ============================================================
@dataclass
class PIController:
    Kp: float
    Ki: float
    integral: float = 0.0
    integral_limit: float = 10.0

    def update(self, error: float, dt: float) -> float:
        p_term = self.Kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.Ki * self.integral
        return p_term + i_term

    def reset(self):
        self.integral = 0.0

class ServoClock:
    def __init__(self, initial_time: float, drift_ppm: float, controller: PIController):
        self.drift = drift_ppm * 1e-6
        self.controller = controller
        self.time = initial_time
        self.rate_correction = 0.0
        self.effective_rate = 1.0 + self.drift

    def update(self, dt: float, target_time: Optional[float] = None):
        if target_time is not None:
            error = target_time - self.time
            self.rate_correction = self.controller.update(error, dt)

        self.effective_rate = 1.0 + self.drift + self.rate_correction
        self.time += dt * self.effective_rate

    def current_time(self) -> float:
        return self.time

class BaselineClock:
    def __init__(self, initial_time: float, drift_ppm: float):
        self.drift = drift_ppm * 1e-6
        self.time = initial_time

    def update(self, dt: float):
        self.time += dt * (1.0 + self.drift)

    def current_time(self) -> float:
        return self.time


# ============================================================
# Synchronization Metrics
# ============================================================
@dataclass
class SyncMetrics:
    mean_offset: float
    mean_abs_offset: float
    max_abs_offset: float
    rms_offset: float
    std_dev: float
    jitter: float
    convergence_time: Optional[float]
    settling_time_1pct: Optional[float]
    steady_state_mean: float
    steady_state_std: float

    @staticmethod
    def compute(offsets: List[float], times: List[float], initial_offset: float) -> Optional['SyncMetrics']:
        if not offsets or len(offsets) < 2:
            return None

        arr = np.array(offsets)
        abs_arr = np.abs(arr)

        steady_idx = max(1, int(len(offsets) * 0.7))
        steady_arr = arr[steady_idx:]

        return SyncMetrics(
            mean_offset=np.mean(arr),
            mean_abs_offset=np.mean(abs_arr),
            max_abs_offset=np.max(abs_arr),
            rms_offset=np.sqrt(np.mean(arr**2)),
            std_dev=np.std(arr),
            jitter=np.std(steady_arr),
            convergence_time=SyncMetrics._find_convergence(times, offsets, 0.001),
            settling_time_1pct=SyncMetrics._find_settling(times, offsets, 1.0, initial_offset),
            steady_state_mean=np.mean(steady_arr),
            steady_state_std=np.std(steady_arr),
        )

    @staticmethod
    def _find_convergence(times: List[float], offsets: List[float], threshold: float) -> Optional[float]:
        for i in range(len(offsets)):
            if all(abs(o) < threshold for o in offsets[i:]):
                return times[i]
        return None

    @staticmethod
    def _find_settling(times: List[float], offsets: List[float], pct: float, initial_offset: float) -> Optional[float]:
        init = abs(initial_offset) if abs(initial_offset) > 1e-9 else 1.0
        band = (pct / 100.0) * init
        for i in range(len(offsets)):
            if all(abs(o) <= band for o in offsets[i:]):
                return times[i]
        return None

    def print_summary(self, title: str, initial_offset: float):
        print(f"\n{'=' * 90}")
        print(f"  {title}")
        print(f"  Initial offset: {abs(initial_offset)*1000:.3f} ms")
        print(f"{'=' * 90}")
        print(f"  ACCURACY (closeness to true time):")
        print(f"    Mean offset (bias):     {self.mean_offset*1000:8.3f} ms")
        print(f"    Mean |offset|:          {self.mean_abs_offset*1000:8.3f} ms")
        print(f"    RMS offset:             {self.rms_offset*1000:8.3f} ms")
        print(f"    Max |offset|:           {self.max_abs_offset*1000:8.3f} ms")
        print(f"  ")
        print(f"  PRECISION (consistency/repeatability):")
        print(f"    Std deviation:          {self.std_dev*1000:8.3f} ms")
        print(f"    Steady-state jitter:    {self.jitter*1000:8.3f} ms")
        print(f"  ")
        print(f"  CONVERGENCE:")
        conv = f"{self.convergence_time:.2f} s" if self.convergence_time else "Not achieved"
        settle = f"{self.settling_time_1pct:.2f} s" if self.settling_time_1pct else "Not achieved"
        print(f"    Time to <1ms:           {conv}")
        print(f"    Settling time (1%):     {settle}")
        print(f"{'=' * 90}\n")


# ============================================================
# Experiment Runner
# ============================================================
class ExperimentRunner:
    def __init__(self, time_source: TimeSource, servo: ServoClock,
                 baseline: BaselineClock, config: dict, name: str, save_path: str):
        self.source = time_source
        self.servo = servo
        self.baseline = baseline
        self.config = config
        self.name = name
        self.save_path = save_path

        self.exp_times: List[float] = []
        self.servo_times: List[float] = []
        self.baseline_times: List[float] = []
        self.sync_times: List[float] = []
        self.offsets: List[float] = []

        self.last_status = time.time()

    def run(self) -> Tuple[Optional[SyncMetrics], List[float], List[float]]:
        print(f"\n{'=' * 90}")
        print(f"  EXPERIMENT: {self.name}")
        print(f"  Protocol: {self.source.name()}")
        print(f"  Duration: {self.config['duration']:.0f}s | Sync: {self.config['sync_period']:.1f}s")
        print(f"  Initial offset: {self.config['initial_offset']:.3f}s | Drift: {self.config['drift_ppm']:.0f} ppm")
        print(f"{'=' * 90}")

        time.sleep(2)

        start = time.time()
        next_sample = start
        next_sync = start + self.config['start_delay']
        prev = start

        while True:
            now = time.time()
            elapsed = now - start

            if elapsed > self.config['duration']:
                break

            dt = now - prev
            prev = now

            target = None
            if elapsed >= self.config['start_delay'] and now >= next_sync:
                target = self.source.query_time()

                if target is not None:
                    offset = target - self.servo.time
                    self.offsets.append(offset)
                    self.sync_times.append(elapsed)

                    print(f"  [t={elapsed:6.1f}s] SYNC  offset={offset*1000:+9.3f}ms  |  correction={self.servo.rate_correction*1e6:+9.3f}ppm")

                next_sync += self.config['sync_period']

            self.servo.update(dt, target)
            self.baseline.update(dt)

            self.exp_times.append(elapsed)
            self.servo_times.append(self.servo.current_time())
            self.baseline_times.append(self.baseline.current_time())

            if now - self.last_status >= self.config.get('status_interval', 10.0):
                print(f"  [t={elapsed:6.1f}s] STATUS  clock={self.servo.time:12.6f}  |  rate={self.servo.effective_rate:.9f}")
                self.last_status = now

            next_sample += self.config['sampling_period']
            sleep = next_sample - time.time()
            if sleep > 0:
                time.sleep(sleep)

        print(f"\n{'=' * 90}")
        print(f"  COMPLETE: {self.name}")
        print(f"  Success rate: {self.source.get_success_rate():.1f}%")
        print(f"  Sync points: {len(self.offsets)}")
        if self.offsets:
            print(f"  Final offset: {self.offsets[-1]*1000:+.3f} ms")
        print(f"{'=' * 90}\n")

        return self._process_results()

    def _process_results(self) -> Tuple[Optional[SyncMetrics], List[float], List[float]]:
        if not self.offsets:
            print(f"  WARNING: No sync points for {self.name}")
            return None, [], []

        metrics = SyncMetrics.compute(self.offsets, self.sync_times, self.config['initial_offset'])

        if metrics:
            metrics.print_summary(f"{self.source.name()} - {self.name}", self.config['initial_offset'])

        df = pd.DataFrame({
            'exp_time': self.exp_times,
            'servo_time': self.servo_times,
            'baseline_time': self.baseline_times,
        })

        df['sync_time'] = None
        df['offset'] = None
        for t, o in zip(self.sync_times, self.offsets):
            idx = (np.abs(np.array(self.exp_times) - t)).argmin()
            df.loc[idx, 'sync_time'] = t
            df.loc[idx, 'offset'] = o

        os.makedirs(self.save_path, exist_ok=True)
        filename = f"{self.source.name()}_{self.name.replace(' ', '_').replace('/', '_')}.csv"
        filepath = os.path.join(self.save_path, filename)
        df.to_csv(filepath, index=False)
        print(f"  Data saved: {filepath}")

        return metrics, self.sync_times, self.offsets


# ============================================================
# Plotting Functions
# ============================================================
def plot_time_evolution(variation, save_path, protocol="NTP"):
    safe_variation = variation.replace(" ", "_").replace("/", "_")
    filename = f"{protocol}_{safe_variation}.csv"
    filepath = os.path.join(save_path, filename)

    if not os.path.exists(filepath):
        print(f" [PLOT] File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Ensure required columns exist
    if not {'exp_time', 'servo_time', 'baseline_time'}.issubset(df.columns):
        print(" [PLOT] Required columns missing in CSV")
        return None

    # Base values
    initial_servo = df['servo_time'].iloc[0]
    initial_offset = 20.0
    initial_ref = initial_servo - initial_offset

    # Relative values
    relative_real = df['exp_time'].to_numpy()
    relative_servo = df['servo_time'].to_numpy() - initial_ref
    relative_baseline = df['baseline_time'].to_numpy() - initial_ref

    # Sync lines (safe check)
    sync_times = []
    if 'sync_time' in df.columns:
        sync_times = df['sync_time'].dropna().to_numpy()

    # Plot
    os.makedirs(save_path, exist_ok=True)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['exp_time'], relative_real, 'r-', label='Real Time (Time Source)')
    plt.plot(df['exp_time'], relative_baseline, 'b--', label='Clock without Synchronization')
    plt.plot(df['exp_time'], relative_servo, 'g-', label='Clock with Synchronization')

    for t in sync_times:
        plt.axvline(t, ls='--', color='gray', alpha=0.3)

    plt.xlabel('Experiment Time (s)')
    plt.ylabel('Relative Time (s)')
    plt.title(f'Time Evolution - {protocol} {variation}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_filename = f"Time_Evolution_{protocol}_{safe_variation}.png"
    plot_filepath = os.path.join(save_path, plot_filename)
    plt.savefig(plot_filepath, dpi=150)

    print(f" [PLOT] Saved: {plot_filename}")
    return fig

def plot_comparison(ntp_data: Tuple[List[float], List[float]],
                    ptp_data: Tuple[List[float], List[float]],
                    var_name: str, save_path: str):
    if not ntp_data or not ptp_data:
        print(f"  [PLOT] Skipping {var_name} - missing data")
        return None

    ntp_t, ntp_off = ntp_data
    ptp_t, ptp_off = ptp_data

    if not ntp_t or not ptp_t:
        print(f"  [PLOT] Skipping {var_name} - empty data")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(ntp_t, [o*1000 for o in ntp_off], label='NTP', alpha=0.7, marker='.', markersize=4)
    ax1.plot(ptp_t, [o*1000 for o in ptp_off], label='PTP', alpha=0.7, marker='.', markersize=4)
    ax1.axhline(0, linestyle='--', alpha=0.3)
    ax1.axhline(1, linestyle=':', alpha=0.3)
    ax1.axhline(-1, linestyle=':', alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Offset (ms)')
    ax1.set_title(f'Offset Evolution - {var_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(ntp_t, [abs(o)*1000 for o in ntp_off], label='NTP', alpha=0.7, marker='.', markersize=4)
    ax2.semilogy(ptp_t, [abs(o)*1000 for o in ptp_off], label='PTP', alpha=0.7, marker='.', markersize=4)
    ax2.axhline(1, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('|Offset| (ms, log scale)')
    ax2.set_title('Convergence Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    filename = f"comparison_{var_name.replace(' ', '_').replace('=', '')}.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()

    return fig

def plot_all_variations(protocol: str, data_dict: dict, save_path: str):
    if not data_dict:
        print(f"  [PLOT] No data for {protocol}")
        return None

    fig = plt.figure(figsize=(14, 8))

    plotted = False
    for name, (times, offsets) in data_dict.items():
        if times and offsets:
            plt.plot(times, [o*1000 for o in offsets], '.-', label=name, alpha=0.8, markersize=3)
            plotted = True

    if not plotted:
        plt.close()
        return None

    plt.axhline(0, linestyle='--', alpha=0.3)
    plt.axhline(1, linestyle=':', alpha=0.3)
    plt.axhline(-1, linestyle=':', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Offset (ms)')
    plt.title(f'{protocol} - All Variations')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    filename = f"{protocol.replace(' ', '_').replace('-', '_')}_all.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()

    return fig

def save_metrics_csv(protocol: str, results: List[Tuple[str, SyncMetrics]], save_path: str):
    os.makedirs(save_path, exist_ok=True)

    headers = [
        "variation",
        "mean_offset_ms",
        "mean_abs_offset_ms",
        "rms_offset_ms",
        "max_offset_ms",
        "std_dev_ms",
        "jitter_ms",
        "convergence_s",
        "settling_1pct_s"
    ]

    rows = []
    for name, m in results:
        rows.append([
            name,
            m.mean_offset * 1000,
            m.mean_abs_offset * 1000,
            m.rms_offset * 1000,
            m.max_abs_offset * 1000,
            m.std_dev * 1000,
            m.jitter * 1000,
            m.convergence_time if m.convergence_time else None,
            m.settling_time_1pct if m.settling_time_1pct else None
        ])

    df = pd.DataFrame(rows, columns=headers)
    filepath = os.path.join(save_path, f"{protocol}_metrics.csv")
    df.to_csv(filepath, index=False)
    print(f"[CSV] Saved metrics: {filepath}")
    return df

def print_metrics_table(protocol_name: str, results: List[Tuple[str, SyncMetrics]]):
    if not results:
        print(f"\n  No results for {protocol_name}\n")
        return

    print(f"\n{'='*130}")
    print(f"  {protocol_name.upper()} - Results Summary")
    print(f"{'='*130}")

    headers = [
        "Variation",
        "Mean (ms)",
        "RMS (ms)",
        "Max (ms)",
        "StdDev (ms)",
        "Jitter (ms)",
        "Conv (s)",
        "Settle (s)"
    ]

    rows = []
    for name, m in results:
        rows.append([
            name,
            f"{m.mean_offset*1000:.3f}",
            f"{m.rms_offset*1000:.3f}",
            f"{m.max_abs_offset*1000:.3f}",
            f"{m.std_dev*1000:.3f}",
            f"{m.jitter*1000:.3f}",
            f"{m.convergence_time:.2f}" if m.convergence_time else "N/A",
            f"{m.settling_time_1pct:.2f}" if m.settling_time_1pct else "N/A"
        ])

    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    print(" | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)))
    print("-" * (sum(col_widths) + len(headers)*3 - 1))
    for row in rows:
        print(" | ".join(f"{v:<{w}}" for v, w in zip(row, col_widths)))
    print(f"{'='*130}\n")

def print_final_summary(ntp_df: pd.DataFrame, ptp_df: pd.DataFrame):
    merged = ntp_df.merge(ptp_df, on="variation", suffixes=("_NTP", "_PTP"))

    print(f"\n{'='*140}")
    print("FINAL SUMMARY: NTP vs PTP")
    print(f"{'='*140}")

    headers = [
        "Variation",
        "Accuracy NTP", "Accuracy PTP", "Better",
        "Precision NTP", "Precision PTP", "Better",
        "Conv NTP", "Conv PTP"
    ]

    rows = []
    for _, r in merged.iterrows():
        better_acc = "NTP" if r['rms_offset_ms_NTP'] < r['rms_offset_ms_PTP'] else "PTP"
        better_prec = "NTP" if r['std_dev_ms_NTP'] < r['std_dev_ms_PTP'] else "PTP"

        rows.append([
            r["variation"],
            f"{r['rms_offset_ms_NTP']:.3f}ms",
            f"{r['rms_offset_ms_PTP']:.3f}ms",
            better_acc,
            f"{r['std_dev_ms_NTP']:.3f}ms",
            f"{r['std_dev_ms_PTP']:.3f}ms",
            better_prec,
            f"{r['convergence_s_NTP']:.2f}s" if not pd.isna(r['convergence_s_NTP']) else "N/A",
            f"{r['convergence_s_PTP']:.2f}s" if not pd.isna(r['convergence_s_PTP']) else "N/A"
        ])

    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    print(" | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)))
    print("-" * (sum(col_widths) + len(headers)*3 - 1))
    for row in rows:
        print(" | ".join(f"{v:<{w}}" for v, w in zip(row, col_widths)))
    print(f"{'='*140}\n")


# ============================================================
# Experiment Execution
# ============================================================
def run_experiments(time_source: TimeSource, base_config: dict,
                   param_sets: List[dict], save_path: str) -> Tuple[List, dict]:
    results = []
    plot_data = {}

    for params in param_sets:
        config = {**base_config, **params}
        name = params.get("description", f"Sync={config['sync_period']:.1f}s")

        print(f"\nPreparing: {name}")

        ref_time = time_source.query_time()
        init_offset = config["initial_offset"]
        init_servo_time = ref_time + init_offset

        controller = PIController(config["Kp"], config["Ki"])
        servo = ServoClock(init_servo_time, config["drift_ppm"], controller)
        baseline = BaselineClock(init_servo_time, config["drift_ppm"])

        runner = ExperimentRunner(
            time_source=time_source,
            servo=servo,
            baseline=baseline,
            config=config,
            name=name,
            save_path=save_path
        )

        metrics, sync_times, offsets = runner.run()
        if metrics:
            results.append((name, metrics))
            plot_data[name] = (sync_times, offsets)

    return results, plot_data

def get_best_ntp_server(fallback: str) -> str:
    try:
        result = subprocess.run(["ntpq", "-p"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise RuntimeError("ntpq failed")

        match = re.search(r'^\*\s*([^\s]+)', result.stdout, re.MULTILINE)
        if match:
            server = match.group(1).strip()
            print(f"- Detected best NTP server: {server}")
            return server

        print("- No best server in ntpq -p, using fallback")
    except Exception as e:
        print(f"- Error reading ntpq: {e}")

    return fallback




# ============================================================
# Main
# ============================================================
def main():
    best_server = get_best_ntp_server(DEFAULT_CONFIG["ntp_pool"][0])
    ntp_pool = [best_server] + [s for s in DEFAULT_CONFIG["ntp_pool"] if s != best_server]
    print(f"Using NTP pool: {ntp_pool[0]} (primary) + {len(ntp_pool)-1} fallbacks\n")

    base_path = os.path.join("Results")


    # EXPERIMENT 1
    if True:
        base_config = {
            "sampling_period": 0.1,
            "duration": 180,
            "drift_ppm": 500.0,
            "initial_offset": 20.0,
            "start_delay": 5.0,
            "offset_noise_std": 200e-6,
        }

        servo_variations = [
            #{"Kp": 0.05, "Ki": 0.025, "sync_period": 10.0, "description": "Conservative PI"},
            #{"Kp": 0.10, "Ki": 0.05,  "sync_period": 10.0, "description": "Balanced PI"},
            #{"Kp": 0.15, "Ki": 0.075, "sync_period": 10.0, "description": "Aggressive PI"},
            {"Kp": 0.05, "Ki": 0.015, "sync_period": 10.0, "description": "Overdamped (Conservative)"},
            {"Kp": 0.07, "Ki": 0.025, "sync_period": 10.0, "description": "Critically damped (Balanced)"},
            {"Kp": 0.12, "Ki": 0.045, "sync_period": 10.0, "description": "Underdamped (Agressive)"},

            {"Kp": 0.10, "Ki": 0.030, "sync_period": 5.0,  "description": "Fast sync (5s, stable)"},
            {"Kp": 0.05, "Ki": 0.015, "sync_period": 20.0, "description": "Slow sync (20s, stable)"},
        ]

        servo_path = os.path.join(base_path, "Exp1_Servo_Tuning")
        ntp_source1 = NTPSource(ntp_pool, base_config["offset_noise_std"],
                            DEFAULT_CONFIG["ntp_timeout"], DEFAULT_CONFIG["max_consecutive_failures"])
        servo_results, servo_plot_data = run_experiments(ntp_source1, base_config, servo_variations, servo_path)
        print_metrics_table("Servo Tuning", servo_results)

        plot_all_variations("Servo Tuning", servo_plot_data, servo_path)
        for var in servo_variations:
            desc = var["description"]
            plot_time_evolution(desc, servo_path, protocol="NTP")

        servo_metrics_path = os.path.join(base_path, "Exp1_Servo_Tuning", "Metrics")
        servo_df = save_metrics_csv("NTP_ServoTuning", servo_results, servo_metrics_path)


    # EXPERIMENT 2
    if False:
        base_config = {
            "sampling_period": 0.1,
            "duration": 300,
            "drift_ppm": 500.0,
            "initial_offset": 20.0,
            "start_delay": 5.0,
            "offset_noise_std": 200e-6,
        }

        comparison_variations = [
            {"Kp": 0.05, "Ki": 0.015, "sync_period": 5.0,  "description": "Sync=5s (Stable) "},
            {"Kp": 0.05, "Ki": 0.015, "sync_period": 10.0, "description": "Sync=10s (Stable)"},
            {"Kp": 0.05, "Ki": 0.015, "sync_period": 20.0, "description": "Sync=20s (Stable)"},
            {"Kp": 0.05, "Ki": 0.015, "sync_period": 30.0, "description": "Sync=30s (Stable)"},
        ]

        if True:
            ntp_path = os.path.join(base_path, "Exp2_NTP_vs_PTP", "NTP")
            ntp_source2 = NTPSource(ntp_pool, base_config["offset_noise_std"],
                                    DEFAULT_CONFIG["ntp_timeout"], DEFAULT_CONFIG["max_consecutive_failures"])
            ntp_results, ntp_plot_data = run_experiments(ntp_source2, base_config, comparison_variations, ntp_path)
            print_metrics_table("NTP", ntp_results)
            plot_all_variations("NTP", ntp_plot_data, ntp_path)

        if True:
            ptp_path = os.path.join(base_path, "Exp2_NTP_vs_PTP", "PTP")
            ptp_source = PTPSource(DEFAULT_CONFIG["ptp_interface"], base_config["offset_noise_std"],
                                DEFAULT_CONFIG["ptp_timeout"], DEFAULT_CONFIG["max_consecutive_failures"])
            ptp_results, ptp_plot_data = run_experiments(ptp_source, base_config, comparison_variations, ptp_path)
            print_metrics_table("PTP", ptp_results)
            plot_all_variations("PTP", ptp_plot_data, ptp_path)

        comparison_path = os.path.join(base_path, "Exp2_NTP_vs_PTP", "Comparisons")
        for var in comparison_variations:
            desc = var["description"]
            if desc in ntp_plot_data and desc in ptp_plot_data:
                plot_comparison(ntp_plot_data[desc], ptp_plot_data[desc], desc, comparison_path)

        metrics_path = os.path.join(base_path, "Exp2_NTP_vs_PTP", "Metrics")
        ntp_df = save_metrics_csv("NTP", ntp_results, metrics_path)
        ptp_df = save_metrics_csv("PTP", ptp_results, metrics_path)
        print_final_summary(ntp_df, ptp_df)


    # Summary
    print(f"\n{'='*90}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
