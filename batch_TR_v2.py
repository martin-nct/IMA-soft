# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:44:04 2024

@author: Martin Nocito
"""

# import os
import numpy as np
import soundfile as sf
# import sounddevice as sd
# import scipy.signal as signal
import funciones_full as ff
from main import Room_IR
 

def save_data(ruta, results, filelist, filt_type):
        
    numcols = len(results[0])
    numparam = len(results[0][0])
    numfiles = int(len(filelist))
    numrows = numparam*numfiles
    
    if len(results) != numfiles:
        raise RuntimeError('numero de archivos y resultados incompatible')
    
    largo = len(max(filelist, key=len)) - 4
    data = np.zeros((numrows+1, numcols+2), dtype=f'<U{largo}')
    
    data[0, 0] = 'File'
    data[0, 1] = 'Freq [Hz]'
    data[0, 2:] = ff.labels_bandas(filt_type)
    namefile = [filelist[i][:-4] for i in range(numfiles) for k in range(numparam)]
    data[1:, 0] = namefile
    data[1:, 1] = numfiles * list(results[0][0].keys())
    
    row = 0
    for file in range(numfiles):
        for param in range(numparam):
            for column in range(2, numcols+2):
                if column != 0 or column != 1:
                    pass
                    data[row+1, column] = (list(results[file][column-2].values())[param])
            row += 1
    
    if ruta != '':
        np.savetxt(ruta, data, fmt='%s', delimiter=',')
        return data

def procesar(filepath, 
             filelist, 
             frecs, 
             filtro, 
             savepath, 
             filterpath=None, 
             rirspath=None):
    
    if filterpath is not None:
        invf, fs_filt= sf.read(filterpath)
    
    results = []
    for i in filelist:
        
        RIR = Room_IR()
        
        # Para convolucionar grabaciones con filtro inverso:
        if filterpath is not None and rirspath is not None:
            RIR.loadRec(filepath + '/' + i)
            RIR.convolveRec(invf, fs_filt)
            RIR.trimIR(RIR.IR, 5)
            sf.write(rirspath + '/' + 'RIR_' + i, RIR.IR, RIR.fs)
        
        # Para importar respuestas ya convolucionadas
        else: 
            RIR.loadIR(filepath + '/' + i)
        
        if filtro == 'octavas': 
            filt_type = 0
        elif filtro == 'tercios': 
            filt_type = 1
        
                
        RIR.filtrarIR(filtro, flip=True)
        
        ETC_set, schroeder_set, SNR_set = RIR.schroederInt()
       
        results_i = RIR.get_acoustical_parameters(schroeder_set, SNR_set)
        
        results.append(results_i)
        print(i)
    
    data = save_data(savepath, results, filelist, filt_type)
    
    return data

#%%
path = 'C:/Users/Fujitsu-A556/Documents/ING SON/IMA/Medicion Teatro San Jose/rirs' #/MED3/SUBJ'
filterpath = None
rirspath = None
filtro = 'tercios'
frecs = ff.f_centrales(filtro)

for med in ['MED3']:
    filepath = path + '/' + med + '/SUBJ'
    
    filelist = ff.batch_read(filepath)
    
    save_folder = filepath
    save_name = 'MED_3_SUBJ'
    
    
    savepath = save_folder + '/' + save_name +'.csv'
    data = procesar(filepath, filelist, frecs, filtro, savepath, filterpath, rirspath)


