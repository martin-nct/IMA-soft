# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:48:00 2024

@author: Martín Nocito
"""

# =============================================================================
# SUBJETIVADOR DE RIRs EN BLOQUE 
# =============================================================================

import numpy as np
import soundfile as sf
from scipy import signal
from time import time
import os

# Introducir la ruta absoluta de la carpeta donde se encuentran las RIRs
path = 'C:/Users/Fujitsu-A556/Documents/ING SON/IMA/Final/soundfield'

# Se requiere crear una carpeta llamada "SUBJ" dentro de esa ruta. 
# Allí se guardan las RIRs subjetivadas

def ventana_subj(fs, tau):
    '''
    Crea la ventana con decaimiento exponencial para la subjetivación de
    señales acústicas. Tau es el tiempo de integración del oído humano.

    Parameters
    ----------
    fs : int
        Frecuencia de muestreo.
    tau : float
        Tau, en milisegundos.

    Returns
    -------
    ventana : array
        Ventana para realizar la subjetivación.

    '''
    
    # Genera la parte rectangular de la ventana
    rectangular = int(0.01 * fs)    # 10 ms en muestras
    rectangular = np.ones(rectangular)
    
    # Genera la parte exponencial de la ventana
    N_tau = int(tau * 0.001 * fs)
    exponencial = np.arange(5 * N_tau) # Duración 5 * tau
    exponencial = np.exp(-exponencial / N_tau) 
    
    ventana = np.concatenate((rectangular, exponencial))
    
    return ventana

def subjetivar(x, ventana):
    '''
    Realiza la subjetivación de la señal x con la ventana de integración provista.

    Parameters
    ----------
    x : array
        Señal a subjetivar.
    ventana : array
        Ventana con decaimiento exponencial para realizar la subjetivación.

    Returns
    -------
    y : array
        Señal subjetivada.

    '''
    y = signal.convolve(x, ventana[::-1], 'valid', 'fft')
    y /= np.max(np.abs(y))
    
    return y

def batch_read(filepath, extension='.wav'):
    '''
    Función auxiliar para la lectura de archivos en bloque.

    Parameters
    ----------
    filepath : str
        ruta a la carpeta que contiene los archivos a leer.
    extension : str, optional
        extensión de los archivos que se desea leer. The default is '.wav'.

    Returns
    -------
    filelist : list
        lista cuyos elementos son str de cada archivo que coincide con la
        extensión especificada.

    '''
    templist = os.listdir(filepath)


    filelist = []
    for i in templist:
        if i.endswith(extension):
            filelist.append(i)
    return filelist

t0 = time()

filelist = batch_read(path, '.wav')
savepath = path + '/SUBJ'
taus = [10, 100, 350] # TAU E, en MILISEGUNDOS

for tau in taus:
    iteracion = 0
    for i in filelist:
        x, fs = sf.read(path + '/' + i)
        
        if iteracion == 0:
            ventana = ventana_subj(fs, tau) 
            #Genera la ventana de integración en el primer ciclo
        
        y = subjetivar(x, ventana)
        try:
            sf.write(savepath + f'/Subj{tau}_' + i, y, fs)
        except:
            raise(RuntimeError(f'No existe la ruta {savepath}'))
        
        iteracion += 1
        # print(f'iteracion {iteracion}')
        
t1 = time()
print(f'tiempo de ejecución = {t1 - t0:.3f} s')