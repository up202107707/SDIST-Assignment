import subprocess
import time
import re

Kp = 0.2
Ki = 0.01

offset_integral = 0.0
rate_ppm = 0.0          # current frequency correction in ppm
virtual_time = time.monotonic()

def get_ptp_offset():
    out = subprocess.check_output(["pmc", "-u", "-b", "0", "GET TIME_STATUS_NP"])
    m = re.search(r'master offset\s+:\s+(-?\d+)', out.decode())
    return int(m.group(1)) if m else 0   # offset in nanoseconds

while True:
    real_now = time.monotonic()
    offset_ns = get_ptp_offset()
    offset = offset_ns * 1e-9

    # PI controller
    offset_integral += offset
    rate_ppm = Kp * offset + Ki * offset_integral

    # Advance virtual clock
    real_elapsed = time.monotonic() - real_now
    virtual_time += real_elapsed * (1.0 + rate_ppm * 1e-6)
    virtual_time += offset   # step elimination (can also be slewed)

    print(f"real={real_now:.6f}  virt={virtual_time:.6f}  offset={offset_ns:6d} ns  rate={rate_ppm:+.3f} ppm")

    time.sleep(1)