#include <iostream>
#include <chrono>
#include <mutex>
#include <thread>


class ServoClock {
public:
    // Constructor: initialize with offset (seconds) and drift (ppm)
    ServoClock(double initialOffset = 0.0, double driftPPM = 0.0)
        : offsetSeconds(initialOffset),
          driftPPM(driftPPM),
          lastUpdate(std::chrono::steady_clock::now())
    {}

    // Get current time of the servo clock (in seconds)
    double now() {
        std::lock_guard<std::mutex> lock(mtx);
        auto current = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = current - lastUpdate;

        // Drift correction: driftPPM = parts per million
        double drift = driftPPM * 1e-6 * elapsed.count();

        double currentTime = offsetSeconds + elapsed.count() + drift;
        return currentTime;
    }

    // Apply a phase correction (seconds)
    void adjustPhase(double correction) {
        std::lock_guard<std::mutex> lock(mtx);
        offsetSeconds += correction;
        lastUpdate = std::chrono::steady_clock::now();
    }

    // Apply a frequency correction (drift adjustment in ppm)
    void adjustFrequency(double freqCorrectionPPM) {
        std::lock_guard<std::mutex> lock(mtx);
        driftPPM += freqCorrectionPPM;
    }

    // Reset the clock to a new offset and drift
    void reset(double newOffset, double newDriftPPM) {
        std::lock_guard<std::mutex> lock(mtx);
        offsetSeconds = newOffset;
        driftPPM = newDriftPPM;
        lastUpdate = std::chrono::steady_clock::now();
    }

private:
    double offsetSeconds;    // current offset in seconds
    double driftPPM;         // drift rate in parts per million
    std::chrono::steady_clock::time_point lastUpdate;  // last update time
    std::mutex mtx;          // thread-safe access
};




#include <thread>
#include <chrono>

int main() {
    ServoClock clk(1.0, 20.0); // 1 second initial offset, 20 ppm drift

    for (int i = 0; i < 10; ++i) {
        std::cout << "ServoClock: " << clk.now() << " s" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Apply a phase correction of -0.5 seconds
    clk.adjustPhase(-0.5);

    // Apply a frequency correction of -5 ppm
    clk.adjustFrequency(-5.0);

    std::cout << "After correction: " << clk.now() << " s" << std::endl;

    return 0;
}



