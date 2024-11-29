# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:36:57 2023

@author: Fujitsu-A556
"""

import numpy as np
import scipy.signal as signal
import soundfile as sf
import funciones_full as ff


class Room_IR():
    def __init__(self):
        
        self.fs = None
        self.IR = None
        self.is_binaural = False
        self.IR_L = None
        self.IR_R = None
        
    def loadIF(self, filterpath):
        self.invf, self.fs_invf = sf.read(filterpath)
        
    def loadRec(self, recpath):
        
        self.rec, self.fs = sf.read(recpath)
        if self.rec.ndim > 1:
            self.is_binaural = True
            self.rec_L = self.rec[:, 0] # Left channel of the IR
            self.rec_R = self.rec[:, 1] # Right channel of the IR
            self.rec = self.rec_L
            
    def convolveRec(self, invf, fs_invf, normalize=True):

        if self.fs != fs_invf:
            raise RuntimeError('Las frecuencias de muestreo no coinciden')
        IR = signal.fftconvolve(self.rec, invf, 'full')
        
        if normalize==True:
            self.IR = IR / np.max(np.abs(IR))
        else: self.IR = IR
        
    def trimIR(self, IR, T_end=None):
        # The start of the impulse is found and the previous samples are discarded.
        if T_end is None:
            T_end = 5 # segs
        length = int(T_end * self.fs)
            
        N_start = np.argmax(np.abs(self.IR))
        N_correc = np.argwhere(np.abs(self.IR)>=0.1) # -20 dB 
        delta = N_start - N_correc[0]
        while delta > 200:          # The start is within 200 samples from the maximum.
            N_correc = N_correc[1:]
            delta = N_start - N_correc[0]
        N_correc = int(N_correc[0])
        IR = self.IR[N_correc:]
        if IR.size > length:
            IR = IR[:length]
        
        self.IR = IR
        
        
    def loadIR(self, file):
        self.IR, self.fs = sf.read(file)
        
        if self.IR.ndim > 1:
            self.is_binaural = True
            self.IR_L = self.IR[:, 0] # Left channel of the IR
            self.IR_R = self.IR[:, 1] # Right channel of the IR
            self.IR = (self.IR_L + self.IR_R) / 2 # Combine both channels
            maxval = max(max(abs(self.IR_L)), max(abs(self.IR_R)))
            self.IR_L /= maxval
            self.IR_R /= maxval
        self.IR /= max(abs(self.IR)) 
            
    def filtrarIR(self, banda, flip=True):
        self.banda = banda
        self.impfilt_set = ff.filtrar_bandas(self.IR, self.fs, banda, flip)
    
    def ETC(self):
        
        temp = np.abs(signal.hilbert(self.impfilt_set, axis=1))
        
        f_c = ff.f_centrales(self.banda)
        
        energy_time_curves = []
        for i in range(len(temp[:, 0])):
            f_i = f_c[i]
            M = int(0.5 * self.fs / f_i)
            if M < 5: 
                M = 5
            if M % 2 == 0: 
                M+=1
            ETC_i = ff.mediamovil_rcsv(temp[i], M)
            
            energy_time_curves.append(ff.a_dB(ETC_i))
        
        self.energy_time_curves = np.array(energy_time_curves)
        return self.energy_time_curves
    
    
    def lundeby(self, Ec, windows=10):  
        # 1)Estimate noise in the last 10% of the signal
        noise = np.mean(Ec[int(0.8 * Ec.size):int(0.98 * Ec.size)])
        
        # Indice del maximo, dentro de la primera mitad
        # i_max = np.argmax(Ec[:int(0.5 * Ec.size)])
        # length = int(0.03*self.fs)     # Cantidad de muestras alrededor del maximo
        # i_0 = max(0, i_max - length // 2)
        # i_1 = i_0 + length

        
        # Nivel de señal en 30 ms entorno al maximo 
        signal = np.max(Ec[:int(0.5 * Ec.size)])
        
        if noise > signal - 10: 
            # print('SNR < 10 dB')
            # el punto de cruce es el final de la curva
            return Ec.size-1, signal, noise
        
        # 2) Linear regression #1 from 0 dB to noise + 5 dB
        t = np.arange(0, Ec.size/self.fs, 1/self.fs)
        
        # Calcula el desvio estandar del ruido
        desv = np.std(Ec[int(0.8 * Ec.size):int(0.98 * Ec.size)])
        
        # Ultimo punto en el cual la señal es mayor al ruido + margen
        i_end = ff.last_where(Ec[:int(0.98 * Ec.size)] > (noise + 2*desv), Ec.size-1)
        
        p = ff.cuad_min(t[:i_end], Ec[:i_end]) # Estima la pendiente
        if p[0] > 0: p[0] = -1e-5 # No debería haber pendientes positivas
        
        reg1 = np.polyval(p, t)     # Evalúa la recta
        
        # Preliminary crossing point is t[i_c]. Si la recta queda totalmente por 
        # debajo del ruido, toma la última muestra
        i_c = ff.last_where(reg1 >= noise, Ec.size-1)
    
        # 3) Moving average filter with 3 to 10 windows for every 10 dB of decay
        samples = -10 * self.fs / p[0]  # Number of samples for a 10 dB decay
        N = int(samples // windows) # windows size
        if N%2 == 0: 
            N += 1
        if N <= 1:
            N = 3
        if N >= Ec.size:
            N = 1920
        # print(N)
        Ec2 = ff.mediamovil_rcsv(Ec, N)
        
        # Estima nuevamente nivel de señal
        signal = np.max(Ec2[:int(0.98 * Ec.size)])
        
        delta = 1   
        it = 0
        while delta > 0.01 and it < 10:
            
            # 4) Noise estimation at a point after tc
            if Ec2[i_c:].size > 1:
                noise = np.mean(Ec2[i_c:]) # Post-crossover noise estimation
            
            
            # Última muestra 5 dB por sobre el ruido
            i_end = ff.last_where(Ec2 >= (noise + 5), False)
            # Última muestra 25 dB por sobre el ruido
            i_start = ff.last_where(Ec2 >= (noise + 25), False)
            
            # Si alguna de las dos condiciones no se cumple devuelve el último
            # Punto de cruce estimado
            if i_end == False or i_start == False:
                # print('SNR < 20 dB')
                break
            
            # 5) Estimate slope from 25 to 5 dB above the noise floor.
            
            p = ff.cuad_min(t[i_start:i_end], Ec2[i_start:i_end])
            reg1 = np.polyval(p, t)
            
            # Estima el nuevo punto de cruce. Si la recta queda por debajo del ruido,
            # Devuelve el último punto de cruce obtenido
            i_c2 = ff.last_where(reg1 >= noise, i_c)
            
            # Delta de tiempo entre el punto de cruce anterior y el obtenido
            # Si delta es mayor a 10 ms realiza otra estimación. Maximo 10 iteraciones
            delta = abs(t[i_c2] - t[i_c])
            i_c = i_c2
            it +=1
    
        return i_c, signal, noise
    
    def schroederInt(self):
        ETC_set = self.ETC()
        SNR_set = []
        schroeder_set = []
        f_centrales = ff.f_centrales(self.banda)
        for i, etc in enumerate(ETC_set):
            
            self.f_c = f_centrales[i]
            crosspoint, *SNR = self.lundeby(etc)
            SNR_set.append(SNR)
            
            IR = self.impfilt_set[i, :crosspoint] ** 2
            
            sch = np.cumsum(IR[::-1])[::-1]
            
            sch /= (np.max(sch))
            sch = np.hstack([sch, np.zeros(etc.size-crosspoint)])
            schroeder_set.append(sch)
            
        schroeder_set = 10 * np.log10(np.fmax(np.array(schroeder_set), 1e-9))
        
        return ETC_set, schroeder_set, SNR_set
    
    def acoustical_parameters (self, smoothed_IR, filtered_IR, SNR, IR_R = None):

          # Dictionary to store the parameters
          d = {
              "SNR":"",
              "RT20":"",
               "RT30":"",
               "EDT":"",
               "C50":"",
               "C80":"",
               "Tt":"",
                "EDTt":""
          }
          
          d['SNR'] = np.round(SNR[0] - SNR[1], 3)
          d["RT20"] = ff.calc_RT20(smoothed_IR, self.fs)
          d["RT30"] = ff.calc_RT30(smoothed_IR, self.fs)
          d["EDT"] = ff.calc_EDT(smoothed_IR, self.fs) 
          d["C50"], d["C80"] = ff.c_parameters(filtered_IR, self.fs)
          d["EDTt"], d['Tt'] = ff.calc_EDTt(filtered_IR, smoothed_IR, self.fs)
          if self.is_binaural:
              d["IACCEARLY"] = ff.calc_IACC_early(filtered_IR, IR_R, self.fs)
          return d
          
    def get_acoustical_parameters(self, schroeder_set, SNR_set):
                
        results = []
        for i, sch in enumerate(schroeder_set):
            results.append(self.acoustical_parameters(
                sch, 
                self.impfilt_set[i], 
                SNR_set[i]))

        return results
