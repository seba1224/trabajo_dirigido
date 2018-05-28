
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ipdb
import scipy.fftpack as fftpack

"""valores de prueba"""
# np.random.seed(10)
duracion = 800
N = 1000
DM_index = 2.0
frec_i = 1.5   # ojo q si f_i y f_fin no son floats no funca
frec_fin = 1.2
width_i = 0.0003
width_fin = 0.003
index_width = 4



def frec_spectrum(x, A, mu, sigma, wn_amp):
    output = A*np.exp(-(x-mu)**2/(2*sigma))+wn_amp*np.random.random_sample(len(x))
    return output


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


def plot_frec(x, t, index, label):
    """index es un parametro de la evolucion del chirp en el tiempo"""
    asd = x[5000*index:5000*index+4096]
    zxc = fftpack.fft(asd)
    n = len(asd)
    xf = np.linspace(0.0, int(1.0/2.0*f_sample), int(n/2))
    plt.plot(xf, 2.0/n * np.abs(zxc[:int(n//2)]), label=label)



def espectrograma(input):
    i = 0
    matrix = np.zeros([460, 2048])
    while(i < 460):
        # asd = x[100*i:100*i+4096]
        print i
        asd = input[50, 100*i:100*i+4096]
        zxc = fftpack.fft(asd)
        n = len(asd)
        matrix[i, :] = 2.0/n * np.abs(zxc[:n//2])
        i = i + 1
    plt.figure()
    plt.imshow(np.transpose(matrix[:, :150]), origin='lower')
    plt.show()




""" la idea es que la transformada de f de exp(a*x**2) es sqrt(pi/a)*exp(-pi*k/a)
y usar que F{f(t)*exp(1j*w0*t)} = F(w-w0)
"""

[a, b, t, peak] = evol_peak(duracion, N, DM_index, frec_i, frec_fin)
ancho = evol_width(a, b, width_i, width_fin, index_width)
frec = np.linspace(0.5, 2.4, len(t))
f_sample = (t[2]-t[1])**-1
