from time import time
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import ipdb

# signals parameters
fs = 2
fftsize = 1024
nspecs = fftsize / 2
siglen_input = fftsize * nspecs
rand_signal = np.random.normal(size=siglen_input)  # gaussian: mu=0, sigma=1

# time-varying fir filter parameter
ntaps = 1001
siglen_output = siglen_input - ntaps + 1


frb_cfreq = np.linspace(0.8,   0.3,  siglen_output)
frb_width = np.linspace(0.005, 0.02, siglen_output)

# apply time-varying fir filter
filt_signal = []
start_time = time()
for i, cfreq, width in zip(range(siglen_output), frb_cfreq, frb_width):
    fir = signal.firwin(ntaps, [cfreq - width/2, cfreq + width/2.0], pass_zero=False, nyq=fs)
    a = np.dot(fir, rand_signal[i:i+ntaps])
    filt_signal.append(a)
print "Time: " + str(time() - start_time)
filt_signal = np.array(filt_signal)

# get spectrogram and plot
f, t, Sxx = signal.spectrogram(filt_signal, nperseg=fftsize, fs=fs)
plt.pcolormesh(t, f, 10*np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()
