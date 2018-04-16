import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Number of samplepoints
N = 6000
# sample spacing
T = 1.0/200
x = np.linspace(0.0, N*T, N)
y1 = np.sin(50.0 * 2.0*np.pi*x)*np.exp(-x**2/(2*1))
y2 = np.sin(50.0 * 2.0*np.pi*x)
y1f = scipy.fftpack.fft(y1)
y2f = scipy.fftpack.fft(y2)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)


fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.plot(xf, 2.0/N * np.abs(y1f[:N//2]), 'r')
ax2.plot(xf, 2.0/N * np.abs(y2f[:N//2]), 'b')
plt.show()
