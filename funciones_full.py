# coding: utf-8


# Funciones útiles

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import soundfile as sf
from numpy.linalg import inv
# from scipy.ndimage import median_filter
import os

# =============================================================================
# PLOTEO
# =============================================================================

def plot_sig(x, t=None, fs=1, title='Señal temporal'):
    """
    Grafica una señal en el dominio temporal. 

    Parameters
    ----------
    x : 1darray
        array de una dimensión que se desea graficar.
    fs : int, opcional
        frecuencia de muestreo de x. Por defecto es 1.
    title : str, opcional
        el título del plot. Por defecto es 'Señal temporal'.

    Returns
    -------
    Figure : 
        Gráfico 2D de x.
    """
    if t is None:
        t = np.linspace(0, x.size/fs, x.size)
    plt.figure(figsize=(16,7))
    plt.plot(t, x)
    plt.title(title)
    plt.xlabel('Tiempo[s]')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.show()

def plot_espectro(x, fs, dB=False, title='Espectro', freq_axis='log'):
    
    freq, X = espectro(x, fs, dB)
    
    fig, ax = plt.subplots()
    
    if freq_axis == 'stem':
        ax.stem(freq, X)
    elif freq_axis == 'log':
        ax.semilogx(freq, X, '-r')
    elif freq_axis == 'lin':
        ax.plot(freq, X, '-r')
    ax.set_title(title)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Amplitud')
    # ax.set_xticks()

def ticks_bandas(tipo='tercios', paso=3):
    '''
    Genera los ticks para un gráfico con eje frecuencial.

    Parameters
    ----------
    tipo : str, optional
        Bandas de octava o tercios de octava. The default is 'tercios'.
    paso : int, optional
        Espaciado entre ticks. The default is 3.

    Returns
    -------
    ticks : list
        Contiene los ticks para el gráfico.

    '''
    if tipo == 'tercios':
        freqs = tercios()
        labels = freqs[::paso]
        ticks = []
        for i in range(len(freqs)):
            if freqs[i] in labels:
                ticks.append(freqs[i])
            else: ticks.append('')
        return ticks
    elif tipo == 'octavas':
        freqs = octavas()
        labels = freqs[::paso]
        ticks = []
        for i in range(len(freqs)):
            if freqs[i] in labels:
                ticks.append(freqs[i])
            else: ticks.append('')
        return ticks
    
def labels_bandas(filtro=0, Hz = False):
    
    if filtro == 0: # Filtro de octavas
        center = ['31.5', '63', '125', '250', '500', '1 k', '2 k', '4 k', '8 k', '16 k']
    if filtro == 1: # Filtro de tercios de octava
        center = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', 
                  '250', '315','400', '500', '630', '800', '1 k', '1.25 k', '1.6 k', 
                  '2 k', '2.5 k', '3.15 k', '4 k', '5 k', '6.3 k', '8 k', '10 k', 
                  '12.5 k', '16 k', '20 k']
    if Hz:
        labels = []
        for i in center:
            labels.append(str(i + 'Hz'))
        return labels
    else:
        return center

def plot_Leq(SPL_bandas):
    # Formato para el eje de frecuencia:
    xticks = ticks_bandas()
    
    fig, ax = plt.subplots(figsize=(10,5))
    
    ax.bar(range(len(SPL_bandas)), SPL_bandas, color='b', edgecolor='k', 
            linewidth=1, tick_label=xticks)
    
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel(r'$L_{eq}$(dB SPL)')
    ax.set_title('Ruido de fondo')
    ax.grid(axis='y')
    

def polosyceros(b,a, lim=1.1):
    '''
    Grafica polos y ceros.
    
    Devuelve un gráfico de polos y ceros y sus valores a partir de los
    coeficientes de la función de transferencia. Los coeficientes deben
    estar ordenados en potencias decrecientes de z.
    
    Parameters
    ----------
    b : array-like, list 
        Coeficientes del numerador de la función de transferencia H(z) ordenados
        en potencias decrecientes de z.
    a : array-like, list
        Coeficientes del denominador de la función de transferencia H(z) ordenados
        en potencias decrecientes de z.
    lim : float, opcional
        límites del gráfico. Por defecto es 1.1. El gráfico tendrá las mismas
        proporciones en ambos ejes.
    
    Returns
    -------
    Figure:
        Gráfico 2D
    Print:
        Polos y ceros.
    '''
    z, p, k = sig.tf2zpk(b, a)
    print('Ceros:', np.around(z,5))
    print('Polos:', np.around(p,5))
    tita = np.linspace(0,2*np.pi, 100)
    x = np.cos(tita)
    y = np.sin(tita)

    plt.figure(figsize=(7,7))
    plt.plot(x, y, '--k')
    for i in z:
        plt.scatter(i.real,i.imag, c='b', marker='o')
    for i in p:
        plt.scatter(i.real, i.imag, c='r', marker='x')
    plt.grid(linestyle='--')
    plt.xlabel('Re{z}')
    plt.ylabel('Im{z}')
    plt.title('Diagrama de Polos y Ceros')
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    plt.show()
    
def plot_impz(b, a = 1, l=100):
    """
    Grafica la respuesta al impulso de una función de transferencia.

    Parameters
    ----------
    b : numpy array o int
        numerador de la función de transferencia en potencias decrecuentes 
        negativas de z        
    a : numpy array o int, optional
        denominador de la función transferencia. The default is 1.
    l : int, optional
        cantidad de muestras del impulso. The default is 100.

    Returns
    -------
    None.

    """
    if type(a)== int: #FIR
        l = len(b)
    else: # IIR
        l = 500
    impulse = np.repeat(0.,l)
    impulse[0] =1.
    x = np.arange(0,l)
    response = sig.lfilter(b, a, impulse)
    plt.plot(x, response); plt.grid()
    plt.ylabel('Amplitud')
    plt.xlabel('n (muestras)')
    plt.title('Respuesta al Impulso')
    
def plot_impz_sos(sos, l=100, filtfilt=True, plot=True):
    '''
    Impulse response of a sos digital filter.

    Parameters
    ----------
    sos : array
        Second Order Section coefficients of the digital filter.
    l : int, optional
        Length for the dirac signal. The default is 100.
    filtfilt : bool, optional
        Wether the filter is forward-backward applied or not. The default is True.
    plot : bool, optional
        If True, plots de impulse response. The default is True.

    Returns
    -------
    response : array
        Impulse response of the filter.

    '''
    impulse = np.repeat(0.,l)
    impulse[0] =1.
    x = np.arange(0,l)
    if filtfilt:
        response = sig.sosfiltfilt(sos, impulse)
    else:
        response = sig.sosfilt(sos, impulse)
    if plot:
        plt.figure()
        plt.plot(x, response)
        plt.grid()
        plt.ylabel('Amplitud')
        plt.xlabel('n (muestras)')
        plt.title('Respuesta al Impulso')
    
    return response

def plot_magyfas(w, H, l=15, a=5):
    """
    Grafica magnitud y fase de una fft.

    Parameters
    ----------
    w : numpy array
        eje de frecuencias.
    H : numpy array
        transformada de Fourier de una señal.
    l : int, optional
        ancho de la figura. The default is 15.
    a : int, optional
        alto de la figura. The default is 5.

    Returns
    -------
    None.

    """
    H_mag = np.abs(H)
    H_fase = np.angle(H)

    plt.figure(1, figsize=(l,a))
    plt.plot(w, H_mag, 'r')
    plt.title('Magnitud de H(w)')
    plt.xlabel('Frecuencia (rad/s)')
    plt.ylabel('Magnitud')
    plt.grid()
    plt.show()

    plt.figure(1, figsize=(l,a))
    plt.plot(w, H_fase, 'g')
    plt.title('Fase de H(w)')
    plt.xlabel('Frecuencia (rad/s)')
    plt.ylabel('Fase (rad/s)')
    plt.grid()
    plt.show()


def plot_stft(STFT, fs, T, title='STFT'):

    '''
    Grafica una STFT.
    
    Devuelve un plot donde el eje horizontal es el tiempo y
    el eje vertical es la frecuencia. La amplitud se representa
    con pseudocolor cmap = 'magma'.
    
    Parameters
    ----------
    STFT: list
        Lista con la STFT a graficar
    fs: int
        Frecuencia de muestreo de la señal
    T: float
        Duración en segundos de la señal
    title: str, opcional
        Título del gráfico. Por defecto "'STFT'"
    '''

    STFT_MAG = np.asarray(np.abs(STFT))
    f = np.linspace(0, fs/2, STFT_MAG.shape[1])
    t = np.linspace(0, T, STFT_MAG.shape[0])
    plt.figure(figsize=(20,10))
    plt.pcolormesh(t, f, STFT_MAG.T, cmap='magma')
    plt.ylabel('Frecuencia (Hz)')
    plt.xlabel('Tiempo (s)')
    plt.title(title)
    plt.show()
    
def plot_stft2(STFT, f, t, title='STFT'):
    
    '''
    Grafica la STFT. Utilizar para STFT calculada con la función de scipy.
    
    Devuelve un plot donde el eje horizontal es el tiempo y
    el eje vertical es la frecuencia. La amplitud se representa
    con pseudocolor cmap = 'magma'.
    
    Parameters
    ----------
    STFT: 2Darray
        Array con la STFT a graficar
    f: array
        Eje frecuencial
    t: array
        Eje temporal
    title: str
        Título del gráfico. Por defecto 'STFT'
    '''
    plt.figure(figsize=(10,5))
    plt.pcolormesh(t, f, np.abs(STFT), cmap='magma')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.show()


# =============================================================================
# FOURIER
# =============================================================================

def espectro(x, fs, dB=False):
    '''
    Transformada de Fourier de x. 

    Parameters
    ----------
    x : array
    fs : int
    dB : bool, optional. The default is False.

    Returns
    -------
    freq : array
    X : array
    '''
    X = np.fft.rfft(x)
    X = np.abs(X)
    X /= max(X)
    if dB==True:
        X = 20*np.log10(X)

    freq = np.fft.rfftfreq(x.size, 1/fs)
    
    return freq, X


def stft(x,fs,largo,solap):
    '''
    Calcula la STFT de una señal.
    
    Subdivide la señal y la multiplica por ventanas de tipo Hann, 
    aplica la FFT a cada ventana y devuelve información tiempo-frecuencia
    de la señal original.
    
    Parameters
    ----------
    x: 1darray
        señal temporal
    fs: int
        frecuencia de sampleo
    largo: float
        duración de cada ventana en segundos
    solap: float
        superposicion entre ventanas en segundos
    
    Returns
    -------
    X: list
        STFT de la señal. 
    '''
    N = int(largo*fs)     # Largo de ventana en muestras
    P = int(solap*fs)      # Paso en muestras
    n = np.arange(N)
    hann = 0.5 - 0.5*np.cos(2*np.pi*n/N)    # Ventana Hann
    X = []
    for i in range(0,x.size-N, P):
        y = x[i:i+N]*hann
        Y = np.fft.rfft(y)
        X.append(Y)
    return X

def ssf(x, fs, largo=0.01, solap=None, silencio=0.3, b=1):
    
    '''
    Reducción de ruido por substracción espectral.
    
    Aplica reducción de ruido por el método de substracción espectral a una señal.
    Requiere que la señal no tenga información útil en el inicio. Procesa la señal
    con reducción de ruido residual.
    
    Parameters
    ----------
    x: 1darray 
        Señal a filtrar
    fs: int 
        Frecuencia de muestreo
    largo: float, opcional
        Longitud en segundos de la ventana para realizar la STFT. Por defecto largo = 0.01 seg
    solap: float, opcional
        El solapamiento en segundos entre ventana y ventana en la STFT. Por defecto es la mitad del largo. Debe ser menor
        al largo.
    silencio: float, opcional
        Tiempo en segundos al inicio de la señal donde se estimará el ruido. Se requiere que no haya información útil.
    b: float, opcional
        Parámetro de sobre-sustracción. Usualmente 1<b<2. Por defecto b=1.
    
    Returns
    -------
    tiempo: array
        Vector de tiempo de la señal
    salida: array
        Señal filtrada
    '''
    
    if solap != None:
        solap = int(solap*fs)    # Pone el paso en muestras
    f, t, X = sig.stft(x, fs, nperseg=int(largo*fs), noverlap=solap)    # Calcula STFT
    X = X.T    # Trasponer la matriz
    
    ventanas = []
    fases = []
    for i in range(len(X)):
        ventanas.append(np.abs(X[i]))
        fases.append(np.angle(X[i]))
    
    ruido = np.array(ventanas[0])
    muestras = int(silencio * fs)    # Cantidad de muestras sin señal útil
    M = int(muestras/len(ventanas[0]))
    
    for i in range(1,M):
        ruido += ventanas[i]
    ruido /= M    # Estimador de ruido
    
    ruido *= b    # Oversubstract
    
    for i in range(len(ventanas)):
        for k in range(len(ventanas[i])):
            if ventanas[i][k] > ruido[k]:
                ventanas[i][k] -= ruido[k]
            else: ventanas[i][k] = 0
    
    # Reducción de ruido residual:
    
    maxres = ventanas[0]            
    for i in range(1,M):
        for k in range(len(ventanas[i])):
            if ventanas[i][k] > maxres[k]:
                maxres[k] = ventanas[i][k]    # Selecciona las magnitudes máximas en el silencio inicial luego de la substracción
    
    for i in range(1,len(ventanas)-1):
        for k in range(len(ventanas[i])):
            if ventanas[i][k] < maxres[k]:
                ventanas[i][k] = min(ventanas[i+1][k], ventanas[i][k], ventanas[i-1][k])
                # Elige el valor mínimo en ventanas adyacentes para esa frecuencia
    
    
    out = []
    for i in range(len(ventanas)):
        out.append(ventanas[i] * np.exp(1j*fases[i]))    # Agregamos la magnitud y la fase
    out = np.asarray(out)
    
    tiempo, salida = sig.istft(out.T,fs, noverlap=solap)     # Antitransformamos
    return tiempo, salida

# =============================================================================
# AJUSTES Y ANÁLISIS
# =============================================================================

def SNR(x, sigma=None, inicio=0, fin=100):
    '''
    Calcula la relación señal a ruido (SNR).
    
    Calcula la SNR de una señal con media 0según una porción
    de señal donde se estima la desviación estandar. Se define como:
    SNR = RD{|x|}/sigma donde RD es el rango dinámico.
    
    Parameters
    ----------
    x : 1darray 
        Señal a calcular SNR.
    sigma : float
        Desviación estandar del ruido. Por defecto es None
    inicio : int, opcional
        index desde el cual se calcula la desviación estándar. Por defecto es 0.
    fin : int, opcional
        index hasta el cual se calcula la desviación estándar. Por defecto es 100.
    
    Returns
    -------
    SNR: float
        Relación señal a ruido de x.    
    '''
    if sigma==None:
        sigma = np.std(x[inicio:fin])
    return np.around((np.max(abs(x)) - np.min(abs(x))) / sigma, 3)

def zero_padding(x, N):
    if x.ndim == 1: # Vector
        ceros = np.zeros(N)
        return np.hstack([x, ceros])
    elif x.ndim == 2 and x.shape[1] == 2: # Señal estereo 
        ceros = np.zeros([N, 2])
        return np.vstack([x, ceros])

def rms(x):
    '''
    Calcula el valor eficaz de una señal.

    Parameters
    ----------
    x : array-like
        Señal a calcular el valor RMS.

    Returns
    -------
    float
        Valor RMS.

    '''
    return np.sqrt(np.mean(x**2))

def rms_ventanas(x, fs, tiempo_ventana):
    
    W = int(tiempo_ventana * fs)
    N = x.size
    x_original = x
    
    if N % W != 0:
        ceros = W - N % W
        x = zero_padding(x, ceros)
        N = x.size
    
    Nvent = N // W
    result = np.zeros(Nvent)
    
    for i in range(Nvent):
        result[i] = rms(x[W*i:W*(i+1)])
    
    result[-1] = rms(x_original[W*(Nvent-1):])
    return result

def cuad_min2(datos_x,datos_y,n=1):
    n += 1
    k = len(datos_x)
    A = (np.ones([k, n]) * datos_x[:, np.newaxis]) ** np.arange(0, n)
    coef = inv(A.T @ A) @ A.T @ datos_y[:,np.newaxis]
    return coef[::-1].T[0,:]

def cuad_min(x, y, n=1):
    A = np.vstack([x, np.ones(len(x))]).T
    p = np.linalg.lstsq(A, y, rcond=None)[0]
    return p

def error_cuadratico_medio(u, v):
    """
    Devuelve el error cuadrático medio entre dos arrays u y v.
    """
    return np.mean((u - v) ** 2)

def last_where(condition, false_val):
    """
    Encuentra el último valor en el cual se cumple una condicion. Si la condición
    es falsa a lo largo de todo el vector, devuelve el valor indicado.
    """
    array = np.argwhere(condition)
    if array.size == 0:
        return false_val
    else:
        return int(array[-1])

# =============================================================================
# FILTRADO
# =============================================================================
    
def suavizado(f, amp, octava):
    """
    Realiza un filtrado por bandas.

    Parameters
    ----------
    f : numpy array
        eje de frecuencias
    amp : numpy array
        amplitudes en dB
    octava : int
        fracción de octava en la cual se realiza el filtrado. p.ej. octava = 3
        resulta en un filtrado de 1/3 de octava. Si es 0, no hay filtrado

    Returns
    -------
    ampsmooth : numpy array
        señal suavizada

    """
    ampsmooth = amp
    if octava != 0:     # Posibilidad de no filtrar con octava = 0
        for n in range(f.size):
            finf = f[n] * 2 ** (-1/(2*octava))  # Calcula frecuencia superior
            fsup = f[n] * 2 ** (1/(2*octava))  # Calcula frecuencia inferior
            
            if finf <= f[0]:
                idxinf = 0
            else:
                idxinf = np.argmin(np.abs(f[:n+1] - finf))  # índice de la frecuencia inferior
            
            if fsup >= f[-1]:
                idxsup = f.size - 1
            else:
                idxsup = np.argmin(np.abs(f[:] - fsup)) # índice de la frecuencia superior
               
            temp = 10 ** (0.1 * amp[idxinf:idxsup+1])   # Suma las presiones en la banda
            ampsmooth[n] = 10 * np.log10(sum(temp) / len(amp[idxinf:idxsup+1])) # Promedio ponderado
    return ampsmooth

def butter_pasabanda(N, flow, fhigh, fs):
    '''Devuelve los coeficientes sos de un filtro butterworth pasabanda.'''
    # Se normaliza la frecuencia en [0, pi]
    wlow = flow / (0.5 * fs)
    whigh = fhigh / (0.5 * fs)
    return sig.butter(N, [wlow, whigh], 'bandpass', output='sos')

def butter_pasabajos(N, fc, fs):
    wc = fc / (0.5 * fs)
    return sig.butter(N, wc, 'low', output='sos')

def butter_pasaaltos(N, fc, fs):
    wc = fc / (0.5 * fs)
    return sig.butter(N, wc, 'high', output='sos')

def filtro_pasabanda(señal, flow, fhigh, fs, N=5, filtfilt=False):
    '''
    Aplica un filtro pasabanda tipo 'sos'.

    Parameters
    ----------
    señal : array 1-D
        señal a filtrar.
    flow : int o float
        Frecuencia de corte inferior.
    fhigh : int o float
        Frecuencia de corte superior.
    fs : int
        Frecuencia de muestreo de señal.
    N : int, optional
        Orden del filtro. The default is 5.

    Returns
    -------
    y : 1darray
        señal filtrada.

    '''
    sos = butter_pasabanda(N, flow, fhigh, fs)
    if filtfilt:
        y = sig.sosfiltfilt(sos, señal)
    else: 
        y = sig.sosfilt(sos, señal)
    return y

def parametro_banda(parametro, señal, fs, flow, fhigh, N=5):
    '''
    Función de alto orden que calcula un parámetro en la banda requerida.

    Parameters
    ----------
    parametro : function
        Función a calcular.
    señal : numpy array
        Señal a procesar.
    fs : int
        Frecuencia de muestreo.
    flow : float, int
        Frecuencia de corte inferior de la banda.
    fhigh : float, int
        Frecuencia de corte superior de la banda..
    N : int, optional
        Orden del filtro. The default is 5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    señal_filt = filtro_pasabanda(señal, flow, fhigh, fs, N)
    return parametro(señal_filt, fs)

def octavas(f0=31.5, normalized=True):
    '''
    Devuelve las frecuencias centrales por bandas de octava.

    Parameters
    ----------
    f0 : float, optional
        frecuencia central de la banda inferior en Hz. The default is 31.5.

    Returns
    -------
    list
        frecuencias centrales de bandas de octava

    '''
    if normalized:
        center = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        return center
    else:
        return [round(f0 * 2 ** i, 1) for i in range(0, 10)]

def tercios(f0=25, normalized=True):
    '''
    Devuelve las frecuencias centrales por tercios de octava.

    Parameters
    ----------
    f0 : float, optional
        frecuencia central de la banda inferior en Hz. The default is 25.

    Returns
    -------
    list
        frecuencias centrales de tercio de octava

    '''
    if normalized:
        center = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 
                  500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 
                  5000, 6300, 8000, 10000, 12500, 16000, 20000]
        return center
    else:
        return [round(f0 * 2 ** (i/3), 1) for i in range(0, 29)]

def f_centrales(banda):
    '''
    Frecuencias centrales normalizadas de bandas de tercio de octava u octava.

    Parameters
    ----------
    banda : str
        'tercios' u 'octavas'.

    Returns
    -------
    f_centrales : list
        Frecuencias centrales normalizadas.

    '''
    if banda == 'tercios':
        f_centrales = tercios()
    elif banda == 'octavas':
        f_centrales = octavas()
    else: raise RuntimeError('"banda" debe ser "tercios" u "octavas"')
    return f_centrales

def filtrar_banda(imp, fs, fc, BW, flip=False):
    '''
    Filtro pasa banda alrededor de una frecuencia central y con determinado
    ancho de banda.

    Parameters
    ----------
    imp : array-type
        Señal a filtrar.
    fs : int
        Frecuencia de muestreo de imp.
    fc : int o float
        Frecuencia central de la banda a filtrar.
    BW : int o float
        Ancho de banda deseado. Por ejemplo, para una octava BW = 1; para
        tercio de octava BW = 3; para 1/48 de octava BW = 48; etc.
    flip : bool, optional
        Si True, se filtra la señal desde el final hacia el inicio. The 
        default is False.

    Returns
    -------
    array
        Señal filtrada.

    '''

    finf = 2 ** (-1/ (2*BW) ) * fc      # Fc inferior
    fsup = 2 ** (1/ (2*BW) ) * fc       # Fc superior
    
    if flip:        # Si se desea filtrar con la señal espejada en eje x
        impflip = np.flip(imp)
        impfilt = filtro_pasabanda(impflip, finf, fsup, fs, 5)
        return np.flip(impfilt)
    else:
        impfilt = filtro_pasabanda(imp, finf, fsup, fs, 5)
        return impfilt

def filtrar_bandas(impulse, fs, banda, flip=False):
    '''
    Filtra una señal en bandas de octava o tercio de octava y devuelve una
    matriz con la señal filtrada en cada una de ellas.

    Parameters
    ----------
    impulse : array-like
        Señal a filtrar.
    fs : int
        Frecuencia de muestreo de impulse.
    banda : str
        'tercios' u 'octavas'.
    flip : bool, optional
        Si True, se filtra la señal desde el final hacia el inicio. The 
        default is False.

    Returns
    -------
    array
        Array de shape = (N, M) para N bandas y M muestras de impulse.

    '''
    impfilt_set = []
    f_c = f_centrales(banda)
    if banda=='tercios': BW = 3
    elif banda=='octavas': BW = 1
        
    for i in f_c:
        impfilt = filtrar_banda(impulse, fs, i, BW, flip)
        impfilt_set.append(impfilt)
    
    return np.array(impfilt_set)

def H_tercios(frecs):
    salida = []
    for i in range(len(frecs)):
        finf = 2 ** (-1/6) * frecs[i]
        fsup = 2 ** (1/6) * frecs[i]
        sos = butter_pasabanda(5, finf, fsup, fs=44100)
        w, h = sig.sosfreqz(sos, worN=16384, fs=44100)
        salida.append(np.abs(h))
    return salida, w

def calcula_parametro(param, imp, fs, filtro='tercios', N=5):
    '''
    Calcula el parámetro 'param' para una señal por octavas o tercios
    de octava.

    Parameters
    ----------
    param : function
        función que calcula el parámetro. Debe ser del tipo f(x, fs).
    imp : 1darray
        señal a procesar. Típicamente una respuesta al impulso.
    fs : int
        Frecuencia de muestreo de imp.
    filtro : str, optional
        Tipo de filtro a aplicar. Si filtro='octavas', calcula por octavas, 
        si es 'tercios', por tercios. The default is 'tercios'.

    Returns
    -------
    salida : list
        lista con los parámetros calculados para cada banda.

    '''
    if filtro == 'tercios':
        salida = []
        frecs = tercios()
        for i in range(len(frecs)):
            finf = 2 ** (-1/6) * frecs[i]
            fsup = 2 ** (1/6) * frecs[i]
            impfilt = filtro_pasabanda(imp, finf, fsup, fs, N)
            salida.append(param(impfilt, fs))
        return salida
    elif filtro == 'octavas' :
        salida = []
        frecs = octavas()
        for i in range(len(frecs)):
            finf = 2 ** (-1/2) * frecs[i]
            fsup = 2 ** (1/2) * frecs[i]
            impfilt = filtro_pasabanda(imp, finf, fsup, fs, N)
            salida.append(param(impfilt, fs))
        return salida

# =============================================================================
# ANDA MEDIO COMO EL OJETE:
# =============================================================================
    
def mediamovil(x,M):
    """
    Aplica un filtro de media móvil de manera recursiva sobre una señal.

    Parámetros de entrada: 

      x:             Señal discreta
      M:             Ancho de ventana
       
    Parámetro de salida:
      
    suavizado:            Señal filtrada resultante
    """
    
    suavizado = np.zeros(len(x)-M+1)    # Vector de salida
    suavizado[0] = np.mean(x[0:M-1])    # Se calcula el primer elemento
    for i in range(1,len(x)-M+1):     # Desde el segundo elemento hasta el último
            suavizado[i] = suavizado[i-1] + (x[i+M-1] - x[i-1])/M   # Aplica la fórmula
    delay = (len(x) - len(suavizado))//2    # Para completar el array de salida
    # y que ambos tengan la misma dimensión para poder plotear y comparar.
    suavizado = np.hstack([suavizado[0:delay], suavizado, suavizado[-delay-1:-1]])
    return suavizado

def mediamovil_rcsv(x, M):
    '''
    Implementación de una media movil recursiva. Arrastra error cuando se usa con
    floats y no ints. 

    Parameters
    ----------
    x : array-like
        Señal a filtrar.
    M : int
        Tamaño de ventana.

    Returns
    -------
    y : 1d-array
        Señal filtrada.

    '''
    if type(M) != int:
        M = int(M)
    
    L = len(x) - M
    y = np.zeros(L)
    
    Acc = np.sum(x[0:M-1])
    y[0] = Acc/M
    
    for i in range(1, L):
        Acc = Acc + x[i+M-1] - x[i-1]
        y[i] = Acc/M
    y = np.hstack([y, y[L-M:]])
    return y

# def mediamovil_rcsv(y, m):
#     """medianf calculates de moving median average of "y" input signal over an "m" sized window.
#     median_filter() function of the scipy library is implemented"""
#     env = np.zeros_like(y)
#     for i in range(len(y)):
#         env[i] = median_filter(y[i], m)
    # return env

# =============================================================================
# CONVERSION A dB    
# =============================================================================
    
def a_Pascal(calibracion, señal):
    '''
    Obtiene una señal en Pascales a partir de una referencia.
    
    Con una señal de referencia 94 dB @ 1 kHz obtenida con un calibrador, 
    es posible conocer la sensibilidad del sistema. A partir de ese valor se
    obtiene la presión que representala amplitud de la señal desconocida.

    Parameters
    ----------
    calibracion : str
        Nombre del archivo de calibración. Es la referencia grabada a 94 dB
        1 k Hz.
    señal : str
        Nombre del archivo de señal desconocida..

    Returns
    -------
    señal_Pa : array
        Señal en pascales.
    sensibilidad : float
        Sensibilidad del sistema.

    '''
    cal, fs1 = sf.read(calibracion)
    señal, fs2 = sf.read(señal)
    sensibilidad = rms(cal)    # eV / Pa
    señal_Pa = señal / sensibilidad     # Pa
    
    return señal_Pa, fs2, sensibilidad

def a_dBSPL(señal, fs):
    '''
    Convierte una señal de presión instantánea a nivel de presión (dB SPL)
    con referencia 20e-6 Pa.

    Parameters
    ----------
    señal : array
        Señal a convertir. Debe estar en pascales.
    fs : int
        Frecuencia de sampleo.

    Returns
    -------
    señal_dBSPL : TYPE
        DESCRIPTION.

    '''
    señal_Pef = rms(señal)
    señal_dBSPL = 20 * (np.log10(señal_Pef) - np.log10(2e-5))
    return señal_dBSPL


def nivel_bandas(datos, f_c, freqs, octava, dB=True):
    '''
    Calcula el nivel de señal por bandas del ancho requerido. 

    Parameters
    ----------
    datos : array-type
        Matriz con las señales a procesar en dB. axis=0 debe ser el eje 
        frecuencial.
    f_c : array-type
        lista con las frecuencias centrales de las bandas de interés.
    freqs : array-type
        Eje frecuencial correspondiente a 'datos'.
    octava : int
        Ancho de banda. octava = 1, bandas de octava. octava = 3, tercios
        de octava...        

    Returns
    -------
    numpy array
        datos procesados con información para las bandas indicadas en f_c.

    '''
    bandas = []
    for i in f_c:
        fsup = i * 2 ** (1/(2*octava))
        finf = i * 2 ** (-1/(2*octava))
        
        idxsup = np.argmin(abs(freqs- fsup))
        idxinf = np.argmin(abs(freqs- finf))
        
        if dB:
            aux = 10 ** (0.1 * datos[idxinf:idxsup+1, :])
            bandas.append(10*np.log10(aux.sum(axis=0)
                                      /len(datos[idxinf:idxsup+1, 1])))
        else:
            nivel = 20*np.log10(sum(
                datos[idxinf:idxsup+1]/len(datos[idxinf:idxsup+1])))
            
            bandas.append(nivel)
    return np.array(bandas)[:,1]

def min_distinto_cero(x):
    minimo = 1
    for i in range(x.size):
        if x[i] < minimo and x[i] != 0:
            minimo = x[i]
    return minimo

def min_distinto_cero2(x):
    
    x = np.abs(x)
    ceros = np.argwhere(x == 0)
    nonceros = np.nonzero(x)[0]
    # idx = 0
    if nonceros.size == 0:
        return x
    else:
        for i in ceros:
            idx = np.argmin(np.abs(nonceros - i))
            x[i] = x[nonceros[idx]]
        return x

def a_dBFS(x):
    x = np.abs(x)
    minimo = min_distinto_cero(x)
    for i in range(x.size):
        if x[i] == 0:
            x[i] = minimo
    return 20 * np.log10(x)

def a_dB2(x):
    x = np.abs(x)
    minimo = min_distinto_cero(x)
    x = np.where(x == 0, minimo, x)
    maximo = np.max(x)
    return 20 * np.log10(x / maximo)

def a_dB(x):
    # x = min_distinto_cero2(x)
    x = np.abs(x)
    x = np.fmax(x, 1e-9)
    maximo = np.max(x)
    return 20 * np.log10(x / maximo)
# =============================================================================
# DESCRIPTORES
# =============================================================================

## RT
def calc_RT20(smoothed_IR, fs):
    # RT20 = 3 * (np.max(np.where(smoothed_IR > -25)) - np.max(np.where(smoothed_IR > -5))) / fs   # Calculate the RT20
    t = np.arange(0, len(smoothed_IR)/fs, 1/fs)
    maxval = np.max(smoothed_IR)
    i_start = int(np.argwhere(smoothed_IR >= maxval -5)[-1])
    i_end = int(np.argwhere(smoothed_IR >= maxval -25)[-1])
    p = cuad_min(t[i_start:i_end], smoothed_IR[i_start:i_end])
    
    return round(-60 / p[0], 3)

def calc_RT30(smoothed_IR, fs):
    # RT30 = 2 * (np.max(np.where(smoothed_IR > -35)) - np.max(np.where(smoothed_IR > -5))) / fs   # Calculate the RT30
    t = np.arange(0, len(smoothed_IR)/fs, 1/fs)
    maxval = np.max(smoothed_IR)
    i_start = int(np.argwhere(smoothed_IR >= maxval -5)[-1])
    i_end = int(np.argwhere(smoothed_IR >= maxval -35)[-1])
    p = cuad_min(t[i_start:i_end], smoothed_IR[i_start:i_end])
    
    return round(-60 / p[0], 3)
    

def calc_EDT(smoothed_IR, fs):
    t = np.arange(0, len(smoothed_IR)/fs, 1/fs)
    maxval = np.max(smoothed_IR)
    i_start = int(np.argwhere(smoothed_IR >= maxval -1)[-1])
    i_end = int(np.argwhere(smoothed_IR >= maxval -10)[-1])
    p = cuad_min(t[i_start:i_end], smoothed_IR[i_start:i_end])

    return round(-60 / p[0], 3)
    
    
## Clarity

def calc_C50(IR, fs):
    t50 = int(0.05 * fs)
    IR = IR ** 2 # Raise the IR to the second power
    C50 = 10 * np.log10(np.cumsum(IR[:t50])  / np.cumsum(IR[t50:])) # Calculate the C50
        
    return round(C50, 3)

def calc_C80(IR, fs):
    t80 = int(0.08 * fs)
    C80 = 10 * np.log10(np.cumsum(IR[:t80])  / np.cumsum(IR[t80:])) # Calculate the C80
    
    return round(C80, 3)

def c_parameters(filtered_IR, fs):
    
    N50 = int(.05 * fs)
    N80 = int(.08 * fs)
    C50 = 10 * np.log10(np.sum(filtered_IR[:N50]**2)  / np.sum(filtered_IR[N50:]**2))
    C80 = 10 * np.log10(np.sum(filtered_IR[:N80]**2)  / np.sum(filtered_IR[N80:]**2))
    return round(C50, 3), round(C80, 3)

## Tt & EDTt

def idx_Tt(filtered_IR):
    '''
    Calcula el índice (muestra) para el Transition Time

    Parameters
    ----------
    filtered_IR : array
        Impulso filtrado en banda.

    Returns
    -------
    index : int

    '''
    # Tt = np.max(np.where(np.cumsum(filtered_IR) <= 0.99 * np.max(np.sum(filtered_IR))))
    
    index = np.argmin(np.cumsum(filtered_IR**2) <= 0.99 * np.sum(filtered_IR**2))
    return index

def calc_EDTt(filtered_IR, smoothed_IR, fs):
    
    index = idx_Tt(filtered_IR) # Transition Time
    Tt = index / fs
    
    peak_idx = np.argmax(filtered_IR)
    
    N = len(smoothed_IR)
    
    t = np.arange(0, N/fs, 1/fs)
    
    p = cuad_min(t[peak_idx : index], smoothed_IR[peak_idx : index])
    
    EDTt = -60 / p[0]
    
    return round(EDTt, 3), round(Tt, 3)

def calc_IACC_early(IR_L, IR_R, fs):

    """
    Calculates the early interaural cross-correlation (IACC) for a stereo impulse response.

    Parameters
    ----------
    IR_L : array-like
        Impulse response of the left channel.
    IR_R : array-like
        Impulse response of the right channel.
    fs : int or float
        Sampling frequency of the impulse response.

    Returns
    -------
    IACC_early : float
        Early interaural cross-correlation (IACC) value.

    """
    
    num = np.correlate(IR_L[0:int(0.08 * fs)], IR_R[0:int(0.08 * fs)], mode='full')
    den = np.sqrt(np.sum(IR_L[0:int(0.08 * fs)] ** 2) * (np.sum(IR_R[0:int(0.08 * fs)] ** 2)))
    IACC_early = np.max(np.abs(num / den)) # Normalizad IACC_early
    
    return round(IACC_early, 3)
# =============================================================================
# PROCESO EN BLOQUES
# =============================================================================

def batch_read(filepath, extension='.wav'):

    templist = os.listdir(filepath)


    filelist = []
    for i in templist:
        if i.endswith(extension):
            filelist.append(i)
    return filelist