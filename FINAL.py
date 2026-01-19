#!/usr/bin/env python3

import subprocess
import socket
import struct
import time
import re
import random
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Configuration constants
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
    "ptp_interface": "enx1e341c7dc4c0",
    "status_interval": 10.0,
}


class TimeSource(ABC):
    # Base class for NTP and PTP with error handling and fallback
    
    def __init__(self, noise_std=0.0, timeout=5.0, max_failures=3):
        self.noise_std = noise_std
        self.timeout = timeout
        self.max_failures = max_failures
        self.consecutive_failures = 0
        self.total_queries = 0
        self.successful_queries = 0
        self.last_good_time = None
        self.last_good_timestamp = None
    
    def query_time(self):
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
    
    def _handle_failure(self, error_msg):
        self.consecutive_failures += 1
        print(f"  [{self.name()}] Query failed ({self.consecutive_failures}/{self.max_failures}): {error_msg}")
        
        if self.consecutive_failures >= self.max_failures:
            print(f"  [{self.name()}] WARNING: Max failures reached, using fallback")
    
    def _get_fallback_time(self):
        if self.last_good_time and self.last_good_timestamp:
            elapsed = time.time() - self.last_good_timestamp
            return self.last_good_time + elapsed
        return time.time()
    
    def get_success_rate(self):
        if self.total_queries == 0:
            return 0.0
        return (self.successful_queries / self.total_queries) * 100
    
    @abstractmethod
    def _query_time_impl(self):
        pass
    
    @abstractmethod
    def name(self):
        pass
    
    def _add_noise(self, value):
        if self.noise_std > 0:
            return value + random.gauss(0, self.noise_std)
        return value

class NTPSource(TimeSource):
    # NTP client with automatic server failover
    
    NTP_DELTA = 2208988800
    NTP_QUERY = b'\x1b' + 47 * b'\0'
    
    def __init__(self, server_pool, noise_std=0.0, timeout=5.0, max_failures=3):
        super().__init__(noise_std, timeout, max_failures)
        self.server_pool = server_pool if isinstance(server_pool, list) else [server_pool]
        self.current_server_idx = 0
        self.server_failures = {s: 0 for s in self.server_pool}
        
        print(f"[NTP] Initialized with {len(self.server_pool)} servers")
        print(f"[NTP] Primary: {self.server_pool[0]}")
    
    def _get_current_server(self):
        min_failures = min(self.server_failures.values())
        best_servers = [s for s, f in self.server_failures.items() if f == min_failures]
        
        if len(best_servers) > 0:
            return best_servers[self.current_server_idx % len(best_servers)]
        return self.server_pool[0]
    
    def _rotate_server(self):
        self.current_server_idx = (self.current_server_idx + 1) % len(self.server_pool)
        print(f"[NTP] Rotating to server: {self._get_current_server()}")
    
    def _query_time_impl(self):
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
                
                # Extract timestamps
                t2 = unpacked[8] + (unpacked[9] / 2**32) - self.NTP_DELTA
                t3 = unpacked[10] + (unpacked[11] / 2**32) - self.NTP_DELTA
                
                # Calculate offset
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
    
    def name(self):
        return "NTP"

class PTPSource(TimeSource):
    # PTP client using pmc (PTP Management Client)
    
    OFFSET_PATTERNS = [
        re.compile(r"master_offset\s+([-+]?\d+)"),
        re.compile(r"master\s+offset\s+([-+]?\d+)"),
        re.compile(r"offset\s+([-+]?\d+)"),
    ]
    
    def __init__(self, interface, noise_std=0.0, timeout=5.0, max_failures=3):
        super().__init__(noise_std, timeout, max_failures)
        self.interface = interface
        self._check_ptp_availability()
    
    def _check_ptp_availability(self):
        try:
            result = subprocess.run(["which", "pmc"], capture_output=True, timeout=2)
            if result.returncode != 0:
                print("[PTP] WARNING: 'pmc' not found. Install linuxptp.")
                print("[PTP] Will use fallback mode (local time)")
        except Exception as e:
            print(f"[PTP] WARNING: Cannot verify pmc: {e}")
    
    def _query_time_impl(self):
        try:
            cmd = ["sudo", "pmc", "-u", "-b", "0", "GET", "TIME_STATUS_NP"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            
            if result.returncode != 0:
                raise RuntimeError(f"pmc failed: {result.stderr.strip()}")
            
            # Debug output for first few failures
            if self.consecutive_failures < 2:
                print(f"[PTP DEBUG] Output:\n{result.stdout}")
            
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
    
    def _parse_offset(self, output):
        for pattern in self.OFFSET_PATTERNS:
            match = pattern.search(output)
            if match:
                return int(match.group(1))
        return None
    
    def name(self):
        return "PTP"


@dataclass
class PIController:
    # Proportional-Integral controller
    Kp: float
    Ki: float
    integral: float = 0.0
    integral_limit: float = 10.0
    
    def update(self, error, dt):
        p_term = self.Kp * error
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.Ki * self.integral
        
        return p_term + i_term
    
    def reset(self):
        self.integral = 0.0

class ServoClock:
    # Clock with PI-controlled frequency correction
    
    def __init__(self, initial_time, drift_ppm, controller):
        self.drift = drift_ppm * 1e-6
        self.controller = controller
        self.time = initial_time
        self.rate_correction = 0.0
        self.effective_rate = 1.0 + self.drift
    
    def update(self, dt, target_time=None):
        if target_time is not None:
            error = target_time - self.time
            self.rate_correction = self.controller.update(error, dt)
        
        self.effective_rate = 1.0 + self.drift + self.rate_correction
        self.time += dt * self.effective_rate
    
    def current_time(self):
        return self.time

class BaselineClock:
    # Unsynchronized clock for comparison
    
    def __init__(self, initial_time, drift_ppm):
        self.drift = drift_ppm * 1e-6
        self.time = initial_time
    
    def update(self, dt):
        self.time += dt * (1.0 + self.drift)
    
    def current_time(self):
        return self.time


@dataclass
class SyncMetrics:
    mean_abs_offset: float
    std_dev: float
    max_abs_offset: float
    min_abs_offset: float
    convergence_time: Optional[float]
    steady_state_offset: float
    jitter: float
    settling_1pct: Optional[float]
    settling_01pct: Optional[float]
    
    @staticmethod
    def compute(offsets, times, initial_offset):
        if not offsets:
            return None
        
        arr = np.array(offsets)
        abs_arr = np.abs(arr)
        steady_idx = int(len(offsets) * 0.7)
        
        return SyncMetrics(
            mean_abs_offset=np.mean(abs_arr),
            std_dev=np.std(arr),
            max_abs_offset=np.max(abs_arr),
            min_abs_offset=np.min(abs_arr),
            convergence_time=SyncMetrics._find_convergence(times, offsets, 0.001),
            steady_state_offset=np.mean(abs_arr[steady_idx:]),
            jitter=np.std(arr[steady_idx:]),
            settling_1pct=SyncMetrics._find_settling(times, offsets, 1.0, initial_offset),
            settling_01pct=SyncMetrics._find_settling(times, offsets, 0.1, initial_offset)
        )
    
    @staticmethod
    def _find_convergence(times, offsets, threshold):
        for i in range(len(offsets)):
            if all(abs(o) < threshold for o in offsets[i:]):
                return times[i]
        return None
    
    @staticmethod
    def _find_settling(times, offsets, pct, initial_offset):
        init = abs(initial_offset) if abs(initial_offset) > 1e-9 else 1.0
        band = (pct / 100.0) * init
        
        for i in range(len(offsets)):
            if all(abs(o) <= band for o in offsets[i:]):
                return times[i]
        return None
    
    def print_summary(self, title, initial_offset):
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"  Initial offset: {abs(initial_offset):.3f} s")
        print(f"{'=' * 80}")
        print(f"  Mean |offset|:       {self.mean_abs_offset*1000:8.3f} ms")
        print(f"  Std deviation:       {self.std_dev*1000:8.3f} ms")
        print(f"  Max |offset|:        {self.max_abs_offset*1000:8.3f} ms")
        print(f"  Steady-state mean:   {self.steady_state_offset*1000:8.3f} ms")
        print(f"  Steady-state jitter: {self.jitter*1000:8.3f} ms")
        
        conv = f"{self.convergence_time:.2f} s" if self.convergence_time else "Not achieved"
        s1 = f"{self.settling_1pct:.2f} s" if self.settling_1pct else "Not achieved"
        s01 = f"{self.settling_01pct:.2f} s" if self.settling_01pct else "Not achieved"
        
        print(f"  Convergence (<1ms):  {conv}")
        print(f"  Settling +-1%:       {s1}")
        print(f"  Settling +-0.1%:     {s01}")
        print(f"{'=' * 80}\n")

class ExperimentRunner:
    
    def __init__(self, time_source, servo, baseline, config, name, save_path):
        self.source = time_source
        self.servo = servo
        self.baseline = baseline
        self.config = config
        self.name = name
        self.save_path = save_path
        
        self.exp_times = []
        self.servo_times = []
        self.baseline_times = []
        self.sync_times = []
        self.offsets = []
        
        self.last_status = time.time()
    
    def run(self):
        print(f"\n{'=' * 80}")
        print(f"  EXPERIMENT: {self.name}")
        print(f"  Protocol: {self.source.name()}")
        print(f"  Duration: {self.config['duration']:.0f}s | Sync: {self.config['sync_period']:.1f}s")
        print(f"  Initial offset: {self.config['initial_offset']:.3f}s | Drift: {self.config['drift_ppm']:.0f} ppm")
        print(f"{'=' * 80}")
        
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
            
            # Synchronization point
            target = None
            if elapsed >= self.config['start_delay'] and now >= next_sync:
                target = self.source.query_time()
                
                if target is not None:
                    offset = target - self.servo.time
                    self.offsets.append(offset)
                    self.sync_times.append(elapsed)
                    
                    print(f"  [t={elapsed:6.1f}s] SYNC  offset={offset:+9.6f}s  |  corr={self.servo.rate_correction:+9.6f}")
                
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
        
        print(f"\n{'=' * 80}")
        print(f"  COMPLETE: {self.name}")
        print(f"  Success rate: {self.source.get_success_rate():.1f}%")
        print(f"  Sync points: {len(self.offsets)}")
        if self.offsets:
            print(f"  Final offset: {self.offsets[-1]:+.6f} s")
        print(f"{'=' * 80}\n")
        
        return self._process_results()
    
    def _process_results(self):
        if not self.offsets:
            print(f"  WARNING: No sync points for {self.name}")
            return None, None, None
        
        metrics = SyncMetrics.compute(self.offsets, self.sync_times, self.config['initial_offset'])
        
        if metrics:
            metrics.print_summary(f"{self.source.name()} - {self.name}", self.config['initial_offset'])
        
        # Save data
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


def plot_comparison(ntp_data, ptp_data, var_name, save_path):
    if not ntp_data or not ptp_data:
        print(f"  [PLOT] Skipping {var_name} - missing data")
        return None
    
    ntp_t, ntp_off = ntp_data
    ptp_t, ptp_off = ptp_data
    
    if not ntp_t or not ptp_t:
        print(f"  [PLOT] Skipping {var_name} - empty data")
        return None
    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ntp_t, [o*1000 for o in ntp_off], 'b.-', label='NTP', alpha=0.7, markersize=3)
    plt.plot(ptp_t, [o*1000 for o in ptp_off], 'r.-', label='PTP', alpha=0.7, markersize=3)
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Offset (ms)')
    plt.title(f'NTP vs PTP - {var_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    filename = f"comparison_{var_name.replace(' ', '_').replace('=', '')}.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150)
    print(f"  [PLOT] Saved: {filename}")
    
    return fig

def plot_all_variations(protocol, data_dict, save_path):
    if not data_dict:
        print(f"  [PLOT] No data for {protocol}")
        return None
    
    fig = plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    plotted = False
    for (name, (times, offsets)), color in zip(data_dict.items(), colors):
        if times and offsets:
            plt.plot(times, [o*1000 for o in offsets], '.-', label=name, color=color, alpha=0.8, markersize=3)
            plotted = True
    
    if not plotted:
        print(f"  [PLOT] No valid data for {protocol}")
        plt.close()
        return None
    
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
    print(f"  [PLOT] Saved: {filename}")
    
    return fig

def print_metrics_table(protocol_name, results):
    if not results:
        print(f"\n  No results for {protocol_name}\n")
        return
    
    print(f"\n{'='*120}")
    print(f"  {protocol_name.upper()} - Results Summary")
    print(f"{'='*120}")
    
    headers = [
        "Variation",
        "Mean |offset| (ms)",
        "Std Dev (ms)",
        "Max |offset| (ms)",
        "Steady (ms)",
        "Jitter (ms)",
        "Conv (s)",
        "Settle 1% (s)",
        "Settle 0.1% (s)"
    ]
    
    rows = []
    for name, m in results:
        conv = f"{m.convergence_time:.2f}" if m.convergence_time else "N/A"
        s1 = f"{m.settling_1pct:.2f}" if m.settling_1pct else "N/A"
        s01 = f"{m.settling_01pct:.2f}" if m.settling_01pct else "N/A"
        
        rows.append([
            name,
            f"{m.mean_abs_offset*1000:.3f}",
            f"{m.std_dev*1000:.3f}",
            f"{m.max_abs_offset*1000:.3f}",
            f"{m.steady_state_offset*1000:.3f}",
            f"{m.jitter*1000:.3f}",
            conv,
            s1,
            s01
        ])
    
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    print(" | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)))
    print("-" * (sum(col_widths) + len(headers)*3 - 1))
    for row in rows:
        print(" | ".join(f"{v:<{w}}" for v, w in zip(row, col_widths)))
    print(f"{'='*120}\n")

def run_experiments(time_source, base_config, param_sets, save_path):
    results = []
    plot_data = {}
    
    for params in param_sets:
        config = {**base_config, **params}
        
        # Use description as key for consistent matching
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

def get_best_ntp_server(fallback):
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




def main():
    base_config = {
        "sampling_period": 0.1,
        "duration": 180.0,
        "drift_ppm": 300.0,
        "initial_offset": 20.0,
        "start_delay": 5.0,
        "offset_noise_std": 200e-6,
        "Kp": 0.10,
        "Ki": 0.05,
        "sync_period": 10.0,
    }
    
    # Detect NTP server
    best_server = get_best_ntp_server(DEFAULT_CONFIG["ntp_pool"][0])
    ntp_pool = [best_server] + [s for s in DEFAULT_CONFIG["ntp_pool"] if s != best_server]
    print(f"Using NTP pool: {ntp_pool[0]} (primary) + {len(ntp_pool)-1} fallbacks\n")
    
    base_path = os.path.join("Results")
    all_figures = []


    # ===================================================================
    # EXPERIMENT 1: Servo Tuning (NTP only)
    # ===================================================================
    if True:
        print("\n" + "=" * 80)
        print("  EXPERIMENT 1: Servo Clock Tuning (NTP)")
        print("=" * 80)
        
        servo_variations = [
            {"Kp": 0.05, "Ki": 0.025, "sync_period": 10.0, "description": "Conservative"},
            {"Kp": 0.10, "Ki": 0.05,  "sync_period": 10.0, "description": "Balanced"},
            {"Kp": 0.15, "Ki": 0.075, "sync_period": 10.0, "description": "Fast"},
            {"Kp": 0.08, "Ki": 0.03,  "sync_period": 5.0,  "description": "Fast sync (5s)"},
            {"Kp": 0.05, "Ki": 0.015, "sync_period": 20.0, "description": "Slow sync (20s)"},
            {"Kp": 0.035,"Ki": 0.008, "sync_period": 30.0, "description": "Very slow sync (30s)"},
        ]

        ntp_servo_path = os.path.join(base_path, "Exp1_NTP_Servo_Tuning")
        ntp_source = NTPSource(ntp_pool, base_config["offset_noise_std"], 
                            DEFAULT_CONFIG["ntp_timeout"], DEFAULT_CONFIG["max_consecutive_failures"])
        
        ntp_servo_results, ntp_servo_plot = run_experiments(ntp_source, base_config, servo_variations, ntp_servo_path)
        print_metrics_table("NTP - Servo Tuning", ntp_servo_results)
        
        fig = plot_all_variations("NTP Servo Tuning", ntp_servo_plot, ntp_servo_path)
        if fig:
            all_figures.append(fig)
    

    # ===================================================================
    # EXPERIMENT 2: NTP vs PTP Protocol Comparison
    # ===================================================================
    if True:
        print("\n" + "=" * 80)
        print("  EXPERIMENT 2: NTP vs PTP Comparison (Same Kp/Ki)")
        print("=" * 80)
        
        comparison_variations = [
            {"Kp": 0.08, "Ki": 0.03,  "sync_period": 5.0,  "description": "Sync=5s"},
            {"Kp": 0.10, "Ki": 0.05,  "sync_period": 10.0, "description": "Sync=10s"},
            {"Kp": 0.05, "Ki": 0.015, "sync_period": 20.0, "description": "Sync=20s"},
            {"Kp": 0.035,"Ki": 0.008, "sync_period": 30.0, "description": "Sync=30s"},
        ]
        
        # NTP experiments
        if True:
            print("\n--- Running NTP ---")
            ntp_comp_path = os.path.join(base_path, "Exp2_NTP_vs_PTP", "NTP")
            ntp_source2 = NTPSource(ntp_pool, base_config["offset_noise_std"],
                                DEFAULT_CONFIG["ntp_timeout"], DEFAULT_CONFIG["max_consecutive_failures"])
            ntp_comp_results, ntp_comp_plot = run_experiments(ntp_source2, base_config, comparison_variations, ntp_comp_path)
            print_metrics_table("NTP - Protocol Comparison", ntp_comp_results)
            
            fig = plot_all_variations("NTP Protocol Comparison", ntp_comp_plot, ntp_comp_path)
            if fig:
                all_figures.append(fig)
        
        # PTP experiments
        if False:
            print("\n--- Running PTP ---")
            ptp_comp_path = os.path.join(base_path, "Exp2_NTP_vs_PTP", "PTP")
            ptp_source = PTPSource(DEFAULT_CONFIG["ptp_interface"], base_config["offset_noise_std"],
                                DEFAULT_CONFIG["ptp_timeout"], DEFAULT_CONFIG["max_consecutive_failures"])
            ptp_comp_results, ptp_comp_plot = run_experiments(ptp_source, base_config, comparison_variations, ptp_comp_path)
            print_metrics_table("PTP - Protocol Comparison", ptp_comp_results)
            
            fig = plot_all_variations("PTP Protocol Comparison", ptp_comp_plot, ptp_comp_path)
            if fig:
                all_figures.append(fig)
            
            # Direct comparisons
            print("\n--- Generating NTP vs PTP comparisons ---")
            comparison_path = os.path.join(base_path, "Exp2_NTP_vs_PTP", "Comparisons")
            for var in comparison_variations:
                desc = var["description"]
                if desc in ntp_comp_plot and desc in ptp_comp_plot:
                    fig = plot_comparison(ntp_comp_plot[desc], ptp_comp_plot[desc], desc, comparison_path)
                    if fig:
                        all_figures.append(fig)
    

    # ===================================================================
    # SUMMARY
    # ===================================================================
    if all_figures:
        print("\nDisplaying all plots (close windows to exit)...")
        plt.show()
    else:
        print("\nNo plots were generated - check for errors above.")


if __name__ == "__main__":
    main()