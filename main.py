import mne
import numpy as np
import time
import matplotlib.pyplot as chrt

# --- config ---
EEG_CHANNELS_ONLY = True
ALPHA_BAND = (8.0, 12.0)
UPDATE_INTERVAL_SECONDS = 0.25 #update interval for the live chart in seconds
THRESHOLD_PERCENTILE = 75

print("--- Starting a live BCI monitoring simulation ---")

# 1.loading and filtering the data
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

if EEG_CHANNELS_ONLY:
    raw.pick_types(eeg=True, meg=False)

raw.filter(l_freq=ALPHA_BAND[0], h_freq=ALPHA_BAND[1], fir_design="firwin")

data, times = raw.get_data(return_times=True)
baseline_threshold = np.percentile(np.abs(data), THRESHOLD_PERCENTILE)
print(f"Calculated threshold for relaxation: {baseline_threshold:.6f}")

fs = raw.info['sfreq']
window_size_samples = int(fs * UPDATE_INTERVAL_SECONDS)

# 2.setting up the live chart
chrt.ion() # turning on the mode for live chart updates
fig, ax = chrt.subplots(figsize=(10, 4))
x_vals, y_vals = [], []
line, = ax.plot([], [], 'b-', lw=2, label="Alpha Power (8-12 Hz)")
ax.axhline(y=baseline_threshold, color='r', linestyle='--', label='Threshold for Activation')

ax.set_ylim(0, baseline_threshold * 2.5)
ax.set_xlim(0, 15)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude / Power")
ax.set_title("Live Monitoring of BCI Alpha Activity")
ax.legend(loc='upper right')

current_state = "OFF"

# 3.simulation loop with live chart updates
for i in range(0, data.shape[1] - window_size_samples, window_size_samples):
    window_data = data[:, i : i + window_size_samples]
    avg_power_in_window = np.mean(np.abs(window_data))
    
    current_time = i / fs
    x_vals.append(current_time)
    y_vals.append(avg_power_in_window)
    
    #line is being updated with new data
    line.set_xdata(x_vals)
    line.set_ydata(y_vals)
    
    #automatically expand the horizontal axis if the current time exceeds the x-axis limit on screen
    if current_time > ax.get_xlim()[1]:
        ax.set_xlim(0, current_time + 10)
        
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # logic for controlling the device
    if avg_power_in_window > baseline_threshold:
        if current_state == "OFF":
            print(f">>> [BRAIN CONTROL] >>> Person has relaxed! DEVICE ON! (Strength: {avg_power_in_window:.6f})")
            current_state = "ON"
    else:
        if current_state == "ON":
            print(jf := f">>> [BRAIN CONTROL] >>> Person has focused. DEVICE OFF! (Strength: {avg_power_in_window:.6f})")
            current_state = "OFF"
            
    time.sleep(0.05)

print("--- Simulation Complete  ---")
chrt.ioff()
chrt.show()