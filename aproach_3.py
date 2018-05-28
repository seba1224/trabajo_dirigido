from time import time
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import ipdb


def evol_peak(duracion, N, DM_index, frec_i, frec_fin):
    """"Entrega un (t,x) deonde t y x son de un  chirp que varia su frecuencia
    en el tiempo, suponiendo que los FRB siguen t=a/(frec**DM)+b"""
    a = duracion/(1.0/frec_fin**DM_index-1.0/frec_i**DM_index)
    b = -a/(frec_i**2)
    t = np.linspace(0, duracion, N)
    peak = np.sqrt(a/(t-b))
    return [a, b, t, peak]


def evol_width(a, b, width_i, width_f, index_width):
    """Entrega distintos valores de width dependiendo del tiempo que se le
    ingrese. Usa que w = c((a/(t-b))**(1/DM_ind))**index_wifth + d
     Antes de usar esta funcion se debe haber sacado a,b de la
    dependencia del DM"""
    # ipdb.set_trace()

    c_var = (width_i - width_f)/((((a/-b)**(1/DM_index))**(-index_width)) -
                                 ((a/(duracion-b))**(1/DM_index))**(-index_width))
    d_var = width_i - c_var*(((a/-b)**(1/DM_index))**(-index_width))
    ancho = c_var*(((a/(t-b))**(1/DM_index))**(-index_width)) + d_var
    return ancho


# Parametros frb
duracion = 800  # duracion = Nsamples/fs; en este caso singlen_input/fs
DM_index = 2.0
frec_i = 0.8  # ojo q si f_i y f_fin no son floats no funca
frec_fin = 0.3    # En el plot se ve q la frec_i y final es la mitad de lo q se coloca...
width_i = 0.0003
width_fin = 0.003
index_width = 4



# signals parameters
fs = 2
fftsize = 1024
nspecs = fftsize / 2
siglen_input = fftsize * nspecs
rand_signal = np.random.normal(size=siglen_input)  # gaussian: mu=0, sigma=1

# time-varying fir filter parameter
ntaps = 1001
siglen_output = siglen_input - ntaps + 1

[a, b, t, peak] = evol_peak(siglen_output/fs, siglen_output, DM_index, frec_i, frec_fin)

frb_cfreq = peak
frb_width = evol_width(a, b, width_i, width_fin, index_width)/100   # Esta funcion ta fallando el width_final esta multiplicado por 100.....


# frb_cfreq = np.linspace(0.8,   0.3,  siglen_output)
#frb_width = np.linspace(0.005, 0.02, siglen_output)

# apply time-varying fir filter
filt_signal = []
start_time = time()
for i, cfreq, width in zip(range(siglen_output), frb_cfreq, frb_width):
    fir = signal.firwin(ntaps, [cfreq - width/2, cfreq + width/2.0], pass_zero=False, nyq=fs)
    aux = np.dot(fir, rand_signal[i:i+ntaps])
    filt_signal.append(aux)
print "Time: " + str(time() - start_time)
filt_signal = np.array(filt_signal)

# get spectrogram and plot
f, t, Sxx = signal.spectrogram(filt_signal, nperseg=fftsize, fs=fs)
plt.pcolormesh(t, f, 10*np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()
