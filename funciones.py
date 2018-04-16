from __future__ import division
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import ipdb


"""valores de prueba"""
np.random.seed(10)
duracion = 800
N = 50000
DM_index = 2.0
frec_i = 1.5   # ojo q si f_i y f_fin no son floats no funca
frec_fin = 1.2
width_i = 0.5
width_fin = 3
index_width = 4


def chirp(duracion, N, DM_index, frec_i, frec_fin, desfase):
    """"Entrega un (t,x) deonde t y x son de un  chirp que varia su frecuencia
    en el tiempo, suponiendo que los FRB siguen t=a/(frec**DM)+b"""
    a = duracion/(1.0/frec_fin**DM_index-1.0/frec_i**DM_index)
    b = -a/(frec_i**2)
    t = np.linspace(0, duracion, N)
    frec = (a/(t-b))**(1/DM_index)
    """para hacer el chirp se usa q phi = phi_0 + int_0^t (2*pi*f(t)), ya que
    f =1/(2pi) * dphi/dt
    Si se quiere ver hay que usar 1/(2*np.pi)*np.gradient(phi, duracion/N)"""
    phi = 2*np.pi*np.cumsum(frec*duracion/N)
    # plt.plot(1/(2*np.pi)*np.gradient(phi, duracion/N)) # mustra la variacion de frec
    # del chirp
    x = np.sin(phi+desfase*t)
    return (a, b, t, x)


def evol_width(a, b, width_i, width_f, t, index_width):
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


prueba = chirp(duracion, N, DM_index, frec_i, frec_fin, 0)
t = prueba[2]
x = prueba[3]
a = prueba[0]
b = prueba[1]
f_sample = (t[2]-t[1])**-1


def plot_frec(x, t, index, label):
    asd = x[5000*index:5000*index+4096]
    zxc = fftpack.fft(asd)
    N = len(asd)
    xf = np.linspace(0.0, 1.0/2.0*f_sample, N/2)
    plt.plot(xf, 2.0/N * np.abs(zxc[:N//2]), label=label)



"""
i = 0
# variacion temporal peak del chirp
plt.figure()
while(i < 9):
    asd = x[5000*i:5000*i+4096]
    zxc = fftpack.fft(asd)
    N = len(asd)
    xf = np.linspace(0.0, 1.0/2.0*f_sample, N/2)
    plt.plot(xf, 2.0/N * np.abs(zxc[:N//2]), label="%d" % (i))
    i = i + 1
plt.show()
plt.legend

# espectrograma hechizo, falta ver a que tiempo corresponde cada iteracion
matrix = np.zeros([460, 2048])
while(i < 460):
    asd = x[100*i:100*i+4096]
    zxc = fftpack.fft(asd)
    N = len(asd)
    matrix[i, :] = 2.0/N * np.abs(zxc[:N//2])
    i = i + 1
plt.figure()
plt.imshow(np.transpose(matrix[:, :150]), origin='lower')
plt.show()


# prueba evolucion temporal
sigma_test = np.zeros(98)
for i in range(0, 98, 1):
    sigma_test[i] = evol_width(a, b, 0.5, 3, t[500*i], index_width)

"""