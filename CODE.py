import subprocess
import socket
import struct
import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Time source abstraction
# ============================================================
class TimeSource(ABC):
    @abstractmethod
    def get_time(self) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class NTPSource(TimeSource):
    OFFSET_REGEX = re.compile(r"offset=([-+]?\d+\.\d+)")

    def get_time(self) -> float:
        try:
            result = subprocess.run(
                ["ntpq", "-c", "rv"],
                capture_output=True,
                text=True,
                timeout=2
            )
            match = self.OFFSET_REGEX.search(result.stdout)
            if match:
                offset = float(match.group(1))
                return time.time() + offset
        except Exception as e:
            print(f"[NTP] Error reading time: {e}")
        return time.time()

    def name(self) -> str:
        return "NTP"


class PTPSource(TimeSource):
    # Multiple regex patterns to handle different pmc output formats
    OFFSET_REGEX_1 = re.compile(r"master_offset\s+([-+]?\d+)")
    OFFSET_REGEX_2 = re.compile(r"master offset\s+([-+]?\d+)")
    OFFSET_REGEX_3 = re.compile(r"offset\s+([-+]?\d+)")

    def get_time(self) -> float:
        try:
            # Query ptp4l for current offset using pmc (needs sudo for socket access)
            result = subprocess.run(
                ["sudo", "pmc", "-u", "-b", "0", "GET TIME_STATUS_NP"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            # Try different regex patterns
            offset_ns = None
            for regex in [self.OFFSET_REGEX_1, self.OFFSET_REGEX_2, self.OFFSET_REGEX_3]:
                match = regex.search(result.stdout)
                if match:
                    offset_ns = int(match.group(1))
                    break
            
            if offset_ns is not None:
                offset_s = offset_ns * 1e-9
                return time.time() + offset_s
            else:
                # Debug: print the actual output to see format
                print(f"[PTP] Could not parse offset from pmc output:")
                print(f"[PTP] Output: {result.stdout[:200]}")
                print(f"[PTP] Stderr: {result.stderr[:200]}")
                
        except FileNotFoundError:
            print("[PTP] Error: 'pmc' command not found. Make sure linuxptp is installed.")
        except subprocess.TimeoutExpired:
            print("[PTP] Timeout querying pmc")
        except Exception as e:
            print(f"[PTP] Error reading time: {e}")
        return time.time()

    def name(self) -> str:
        return "PTP"



# ============================================================
# PI Controller for rate correction
# ============================================================
@dataclass
class PIController:
    Kp: float
    Ki: float
    integral: float = 0.0

    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        correction = self.Kp * error + self.Ki * self.integral
        return correction



# ============================================================
# Servo Clock with rate correction
# ============================================================
class ServoClock:
    def __init__(self, initial_time: float, drift_ppm: float, controller: PIController):
        self.drift = drift_ppm * 1e-6
        self.controller = controller
        self.time = initial_time
        self.rate_correction = 0.0
        self.error_history: List[float] = []

    def update(self, dt: float, target_time: float = None):
        if target_time is not None:
            error = target_time - self.time
            self.rate_correction = self.controller.update(error, dt)
            self.error_history.append(error)

        self.time += dt * (1.0 + self.drift + self.rate_correction)

    def current_time(self) -> float:
        return self.time



# ============================================================
# Baseline clock without corrections (for comparison)
# ============================================================
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
    mean_abs_offset: float
    std_dev: float
    max_abs_offset: float
    min_abs_offset: float
    convergence_time: Optional[float]
    steady_state_offset: float
    jitter: float
    
    @staticmethod
    def compute(offsets: List[float], sync_times: List[float], threshold: float = 0.001) -> 'SyncMetrics':
        offsets_array = np.array(offsets)
        abs_offsets = np.abs(offsets_array)
        
        # Basic statistics
        mean_abs_offset = np.mean(abs_offsets)
        std_dev = np.std(offsets_array)
        max_abs_offset = np.max(abs_offsets)
        min_abs_offset = np.min(abs_offsets)
        
        # Convergence time: time to reach and stay within threshold
        convergence_time = None
        for i in range(len(offsets)):
            if all(abs(offsets[j]) < threshold for j in range(i, len(offsets))):
                convergence_time = sync_times[i]
                break
        
        # Steady-state offset (last 30% of measurements)
        steady_start = int(len(offsets) * 0.7)
        steady_state_offset = np.mean(np.abs(offsets_array[steady_start:]))
        
        # Jitter: standard deviation in steady state
        jitter = np.std(offsets_array[steady_start:])
        
        return SyncMetrics(
            mean_abs_offset=mean_abs_offset,
            std_dev=std_dev,
            max_abs_offset=max_abs_offset,
            min_abs_offset=min_abs_offset,
            convergence_time=convergence_time,
            steady_state_offset=steady_state_offset,
            jitter=jitter
        )
    
    def print_summary(self, protocol_name: str):
        print(f"\n{'='*70}")
        print(f"  {protocol_name} - Synchronization Quality Metrics")
        print(f"{'='*70}")
        print(f"  Mean |Offset|:        {self.mean_abs_offset*1000:.3f} ms")
        print(f"  Std Deviation:        {self.std_dev*1000:.3f} ms")
        print(f"  Max |Offset|:         {self.max_abs_offset*1000:.3f} ms")
        print(f"  Min |Offset|:         {self.min_abs_offset*1000:.3f} ms")
        print(f"  Steady-State Offset:  {self.steady_state_offset*1000:.3f} ms")
        print(f"  Jitter (steady):      {self.jitter*1000:.3f} ms")
        if self.convergence_time is not None:
            print(f"  Convergence Time:     {self.convergence_time:.2f} s")
        else:
            print(f"  Convergence Time:     Not achieved (>1ms threshold)")
        print(f"{'='*70}\n")



# ============================================================
# Experiment Runner with delayed sync start
# ============================================================
class ExperimentRunner:
    def __init__(self, time_source: TimeSource, servo: ServoClock,
                 sampling_period: float, sync_period: float, duration: float,
                 start_delay: float, baseline: BaselineClock, experiment_name: str):
        self.time_source = time_source
        self.servo = servo
        self.baseline = baseline
        self.sampling_period = sampling_period
        self.sync_period = sync_period
        self.duration = duration
        self.start_delay = start_delay
        self.experiment_name = experiment_name

        # Data storage
        self.exp_times: List[float] = []
        self.servo_times: List[float] = []
        self.baseline_times: List[float] = []
        self.sync_times: List[float] = []
        self.sync_exp_times: List[float] = []
        self.target_times: List[float] = []
        self.offsets: List[float] = []

    def run(self):
        print(f"Starting experiment using {self.time_source.name()} source")
        print(f"Duration: {self.duration}s | Sampling: {self.sampling_period}s | Sync every: {self.sync_period}s | Sync starts after: {self.start_delay}s")
        print(f"Initial Servo Time: {self.servo.time:.3f}s")
        print(f"Initial Baseline Time: {self.baseline.time:.3f}s\n")

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

            # Sync only after initial delay
            if elapsed >= self.start_delay and now >= next_sync:
                target_time = self.time_source.get_time()
                offset = target_time - self.servo.time
                self.offsets.append(offset)
                self.sync_times.append(target_time)
                self.sync_exp_times.append(elapsed)
                self.target_times.append(target_time)
                next_sync += self.sync_period
                print(f"[t={elapsed:.3f}s] -> SYNC | Servo: {self.servo.time:.3f} | Target: {target_time:.3f} | Offset: {offset:.6f}s")

            # Update servo and baseline clocks
            self.servo.update(dt=dt, target_time=target_time)
            self.baseline.update(dt=dt)

            # Store data
            self.exp_times.append(elapsed)
            self.servo_times.append(self.servo.time)
            self.baseline_times.append(self.baseline.time)

            if target_time is None:
                print(f"[t={elapsed:.3f}s] -> Servo: {self.servo.time:.3f} | Baseline: {self.baseline.time:.3f}")

            # Wait for next sample precisely
            next_sample += self.sampling_period
            sleep_time = next_sample - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("\nExperiment finished.")
        
        # Compute and display metrics
        if self.offsets:
            metrics = SyncMetrics.compute(self.offsets, self.sync_exp_times)
            metrics.print_summary(f"{self.time_source.name()} Protocol")
            self.plot_results(metrics)
        else:
            print("No synchronization data collected.")

    def plot_results(self, metrics: SyncMetrics):
        # Create new figure with unique number to allow multiple windows
        fig = plt.figure(figsize=(10, 8))
        fig.canvas.manager.set_window_title(self.experiment_name)
        
        # Create 3 subplots
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2)
        ax3 = plt.subplot(3, 1, 3)
        
        # ===== Plot 1: Clock Times =====
        ax1.plot(self.exp_times, self.servo_times, label="Servo Clock (corrected)", 
                color="red", linewidth=1.5)
        ax1.plot(self.exp_times, self.baseline_times, label="Baseline Clock (no corrections)", 
                 color="orange", linestyle="--", linewidth=1.5, alpha=0.7)
        
        # Plot reference time line connecting sync points
        if self.sync_exp_times and self.target_times:
            ax1.plot(self.sync_exp_times, self.target_times, color="green", 
                    linewidth=2, alpha=0.6, label="Reference Time (Target)", zorder=4)
        
        # Plot sync points as dots
        ax1.scatter(self.sync_exp_times, self.target_times, color="green", s=80, 
                   zorder=5, marker='o', edgecolors='darkgreen', linewidths=1.5)
        
        # Plot sync vertical lines
        for sync_time in self.sync_exp_times:
            ax1.axvline(sync_time, color="green", linestyle=":", alpha=0.3, linewidth=1)
        
        ax1.set_xlabel("Experiment Time (s)", fontsize=11)
        ax1.set_ylabel("Clock Time (s)", fontsize=11)
        ax1.set_title(f"{self.experiment_name} - Clock Time Comparison", fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # ===== Plot 2: Offset Evolution =====
        ax2.plot(self.sync_exp_times, [o*1000 for o in self.offsets], color="purple", 
                linewidth=2, marker='o', markersize=6, label="Offset to Target")
        
        # Plot sync vertical lines
        for sync_time in self.sync_exp_times:
            ax2.axvline(sync_time, color="green", linestyle=":", alpha=0.3, linewidth=1)
        
        # Mark convergence time if achieved
        if metrics.convergence_time is not None:
            ax2.axvline(metrics.convergence_time, color="blue", linestyle="--", 
                       linewidth=2, alpha=0.7, label=f"Convergence ({metrics.convergence_time:.1f}s)")
        
        # Add threshold lines
        ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label="±1ms threshold")
        ax2.axhline(-1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2.set_xlabel("Experiment Time (s)", fontsize=11)
        ax2.set_ylabel("Offset to Target (ms)", fontsize=11)
        ax2.set_title("Synchronization Offset Evolution", fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # ===== Plot 3: Convergence Analysis =====
        abs_offsets_ms = [abs(o)*1000 for o in self.offsets]
        ax3.semilogy(self.sync_exp_times, abs_offsets_ms, color="darkblue", 
                     linewidth=2, marker='s', markersize=5, label="|Offset| (log scale)")
        
        # Plot sync vertical lines
        for sync_time in self.sync_exp_times:
            ax3.axvline(sync_time, color="green", linestyle=":", alpha=0.3, linewidth=1)
        
        # Mark convergence time
        if metrics.convergence_time is not None:
            ax3.axvline(metrics.convergence_time, color="blue", linestyle="--", 
                       linewidth=2, alpha=0.7)
        
        # Add threshold line
        ax3.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label="1ms threshold")
        
        # Add metrics text box
        metrics_text = (f"Convergence: {metrics.convergence_time:.1f}s\n" if metrics.convergence_time 
                       else "Convergence: Not achieved\n")
        metrics_text += f"Steady-State: {metrics.steady_state_offset*1000:.3f}ms\n"
        metrics_text += f"Jitter: {metrics.jitter*1000:.3f}ms"
        
        ax3.set_xlabel("Experiment Time (s)", fontsize=11)
        ax3.set_ylabel("|Offset| (ms, log scale)", fontsize=11)
        ax3.set_title("Convergence Analysis", fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.show(block=False) 




# ============================================================
# Main
# ============================================================

def run_single_experiment(time_source: TimeSource, config: dict, experiment_name: str):
    print(f"\n{'='*70}")
    print(f"  {experiment_name}")
    print(f"{'='*70}\n")
    
    # Fetch initial reference time from synchronization source
    print("Fetching initial reference time from synchronization source...")
    reference_time = time_source.get_time()
    initial_servo_time = reference_time + config["initial_offset"]
    print(f"Reference Time: {reference_time:.3f}s")
    print(f"Initial Servo Time (with offset): {initial_servo_time:.3f}s\n")

    # Initialize clocks and controller
    controller = PIController(Kp=config["Kp"], Ki=config["Ki"])
    servo = ServoClock(
        initial_time=initial_servo_time,
        drift_ppm=config["drift_ppm"],
        controller=controller
    )

    # Create baseline clock with same initial conditions
    baseline = BaselineClock(
        initial_time=initial_servo_time,
        drift_ppm=config["drift_ppm"]
    )

    # Create and run experiment
    experiment = ExperimentRunner(
        time_source=time_source,
        servo=servo,
        sampling_period=config["sampling_period"],
        sync_period=config["sync_period"],
        duration=config["duration"],
        start_delay=config["start_delay"],
        baseline=baseline,
        experiment_name=experiment_name
    )

    experiment.run()


def main():
    # Configuration parameters (shared by all experiments)
    config = {
        "sampling_period": 0.1,      # Sampling interval in seconds
        "sync_period": 5.0,           # Time between sync operations in seconds
        "duration": 60.0,             # Total experiment duration in seconds
        "drift_ppm": 300.0,           # Clock drift in parts per million
        "initial_offset": 20.0,       # Initial time offset in seconds
        "Kp": 0.1,                    # Proportional gain
        "Ki": 0.05,                   # Integral gain
        "start_delay": 5.0,           # Delay before first sync in seconds
    }

    # Experiment 1: NTP synchronization
    ntp_source = NTPSource()
    #run_single_experiment(ntp_source, config, "EXPERIMENT 1: NTP Synchronization")

    # Experiment 2: PTP synchronization
    ptp_source = PTPSource()
    run_single_experiment(ptp_source, config, "EXPERIMENT 2: PTP Synchronization")

    print(f"\n{'='*70}")
    print("  All experiments completed!")
    print(f"  Close all plot windows to exit.")
    print(f"{'='*70}\n")
    
    # Keep plots open
    plt.show()


if __name__ == "__main__":
    main()