import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import ipdb


def coef_t_de_f(duracion, N, DM_index, frec_i, frec_f):
    """Toma t=a*f**-DM + b
    N = numero de intervalos temporales.
    """
