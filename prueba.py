import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ipdb


np.random.seed(10)
"""Ojo q si se hace en GHz hay q tomar varios ptos xq o sino no se ve bien la
forma y si dps hay q usar la ifft queda la caga"""


def frec_spectrum(x, A, mu, sigma, wn_amp):
    output = A*np.exp(-(x-mu)**2/(2*sigma))+wn_amp*np.random.random_sample(len(x))
    return output


x = np.linspace(0.5, 2.4, 1000)
plt.clf()
plt.plot(x, frec_spectrum(x, 1, 1.5, 0.0005, 0.1))
plt.xlabel("f[GHz]")
plt.show()

"""
def spectrogram(x, A, mu, sigma, wn_amp, DM_index, DM_amp, ISS_index, ISS_amp,
                delta_t, duracion):
"""


duracion = 800
delta_t = 1000
DM_index = 2
frec_i = 1.5
frec_fin = 1.2
"""tomo t=a*f**-2 + b , la idea es dps hacerlo mas gral"""
a = duracion/(1/frec_fin**DM_index-1/frec_i**DM_index)
b = -a/(frec_i**2)

t = np.linspace(0, duracion, delta_t)


def peak_de_frec(a, b, t):
    """Entrega el valor del peak de frec para un t dado"""
    out = np.sqrt(a/(t-b))
    return out


plt.clf()
plt.plot(t, peak_de_frec(a, b, t))
plt.show()

matrix = np.zeros([len(x), len(t)])
for i in range(0, len(t)):
    matrix[:, i] = frec_spectrum(x, 1, peak_de_frec(a, b, t[i]), 0.0005, 0.1)


plt.imshow(matrix, origin='bottom')  # aca el x es t y el y es frec (no cache como arreglar los ejes)
plt.show()
