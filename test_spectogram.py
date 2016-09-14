from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
freq = np.linspace(1e3, 2e3, N)
x = 0 * np.sin(2*np.pi*freq*time) + 1500

f, t, Sxx = signal.spectrogram(x, fs, scaling =  'spectrum',detrend = False)
print np.max(Sxx)

plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

