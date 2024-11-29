"""
Python implementation of autocorrelation function (ACF) parameters based on the MATLAB codes 
by Sato et al. [the 137th AES convention, Los Angeles, 9181 (2014)].

The initial parameters of the ACF calculation (lines 38-47) should be adjusted according to 
your research requirements.

Note: The last value 'Corr' in the Excel output is a sub-parameter for the tau-e calculation, 
representing the accuracy of the ACF envelope regression (correlation between the ACF envelope
and regression line). If 'Corr' is less than 0.85, the value of tau-e may be questionable.

@author: Shin-ichi Sato (as of April 2, 2024)

"""

import numpy as np
from numpy import pi, polymul
from scipy.signal import bilinear, lfilter
import soundfile as sf
import funciones_full as ff


def process(filepath, int_val=1.0):
    
    
    y, fs = sf.read(filepath)
    
    normalized_y = y / 32767
    file = filepath
    
    if len(normalized_y.shape) == 1:
        y_mono = normalized_y        ### Monaural file
    else:
        y_mono = normalized_y[:, 0]  ### 1st chamnnel of a stereo file
        
    tv = np.arange(len(y_mono)) / fs ### time axis per sample
    
    ### Initial setting parameters
    # int_val  = 1.0  # [s] Temporal window length (ACF size)
    runstep  = 0.1  # [s] Running step (Hop size)
    tmax     = 0.2  # [s] Maximum delay time of the ACF
    mintau1  = 0    # [s] Lower delay time to find the ACF peak for tau1
    maxtau1  = 0.02 # [s] Upper delay time to find the ACF peak for tau1
    defwphi0 = 0.5  # Normalized ACF amplitude to calculate WPhi(0)
    
    ### A-weighting filter (aflt == 1)
    aflt = 1
    
    if aflt == 1:   
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        A1000 = 1.9997
        NUMs = [(2*pi*f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
        DENs = polymul([1, 4*pi*f4, (2*pi*f4)**2],
                       [1, 4*pi*f1, (2*pi*f1)**2])
        DENs = polymul(polymul(DENs, [1, 2*pi*f3]),
                                     [1, 2*pi*f2])
        B, A = bilinear(NUMs, DENs, fs)
        yfilt = lfilter(B, A, y_mono)
    else:
        yfilt = y_mono
    
    ### Correlogram
    dl = int((int_val + tmax) * fs)
    noverlap = dl - int(runstep * fs)
    ncol = int((len(yfilt) - noverlap) / (dl - noverlap))
    colindex = np.arange(ncol) * (dl - noverlap)
    rowindex = np.arange(1, dl + 1)
    taxis = (colindex) / fs
    
    dat = np.zeros((dl, ncol))
    dat[:, :] = yfilt[rowindex[:, np.newaxis] + colindex[np.newaxis, :] - 1]
    for i in range(ncol):
        dat[:, i] = dat[:, i] - np.mean(dat[:, i])
    
    dat2 = dat.copy()
    dat2[int(fs*int_val)-1:, :] = 0
    fft_dat = np.fft.fft(dat, axis = 0)
    fft_dat2_conj = np.conj(np.fft.fft(dat2, axis = 0))
    cor2_freq_domain = fft_dat * fft_dat2_conj
    cor2 = np.real(np.fft.ifft(cor2_freq_domain, axis = 0))
    cor2 = cor2[:int(tmax*fs), :]
    
    normalizer = np.zeros((int(tmax*fs), ncol))
    n = 0
    lag = 0
    tmp1 = dat[:int(fs*int_val), :]
    tmp2 = dat[lag:int(fs*int_val)+lag, :]
    tmp = tmp1 * tmp2 
    Phi_tmp = np.sum(tmp, axis=0)
    Phi_t = np.sum(tmp, axis=0)
    normalizer[n, :] = np.sqrt(Phi_t * Phi_tmp)
    n += 1
        
    for lag in range(1, int(tmax*fs)-1):
        Phi_tmp -= dat[lag, :]**2
        Phi_tmp += dat[int(fs*int_val)+lag, :]**2
        normalizer[n, :] = np.sqrt(Phi_t * Phi_tmp)
        n += 1
    
    cor2 /= normalizer
    cor = cor2
    tauaxis = np.arange(cor.shape[0]) / fs
    
    ### Running ACF parameters
    RunningFactor = np.zeros((len(taxis), 7))
    
    for z in range(len(taxis)):
        acfplot = np.column_stack((tauaxis, cor[:, z]))
    
        ### Phi(0)
        Phi0 = 10 * np.log10(normalizer[0, z])
        
        ### tau1 and phi1
        for j in range(int(tmax*fs)):
            if acfplot[j, 1] < 0:
                tg = j
                break
        if int(mintau1*fs) >= tg:
            tg = int(mintau1*fs)
        acfplot2 = np.zeros(len(acfplot))
        for k in range(len(acfplot)):
            acfplot2[k] = acfplot[k, 1] * np.exp(-(k+1) / (0.01 * fs))
        i_tau = np.argmax(acfplot2[tg:int(maxtau1*fs)])
        tau1 = acfplot[i_tau+tg, 0]
        phi1 = acfplot[i_tau+tg, 1]
        
        ### Wphi(0)
        for l in range(int(tmax*fs)):
            if acfplot[l, 1] < defwphi0:
                tg2 = l
                break
        width = np.interp(defwphi0, [acfplot[tg2,1],acfplot[tg2-1,1]], [acfplot[tg2, 0],acfplot[tg2-1, 0]])
    
        ### taue
        if phi1 <= 0.1:
            for n in range(len(acfplot)):
                if acfplot[n, 1] < 0.1:
                    taue = acfplot[n, 0]
                    break
                    Delta = 0
                    corr = 0
        else:
            lgacfplot = np.column_stack((acfplot[:, 0], 10 * np.log10(np.abs(acfplot[:, 1]))))
    
            #taue_03
            p_taue = np.zeros((10, 2))
            for m in range(1,10):
                start_index = int((m) * 0.003 * fs )
                end_index = int((m+1) * 0.003 * fs )
                r_taue = lgacfplot[start_index:end_index, :]
                ii = np.argmax(r_taue[:, 1])
                p_taue[m, 1] = r_taue[ii, 1]
                p_taue[m, 0] = r_taue[ii, 0] 
            index_taue = np.where(p_taue[:, 0] != 0 )[0]
            rr = p_taue[index_taue,:]
            reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
            taue_03 = 1000 * (-10 - reg0[1]) / reg0[0]
            cor_03 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
            if cor_03.size == 1:
                cor_03 = np.array([[cor_03, 0.0]])
                
            #taue_04
            p_taue = np.zeros((10, 2))
            for m in range(1, 10):
                start_index = int((m) * 0.004 * fs )
                end_index = int((m+1) * 0.004 * fs )
                r_taue = lgacfplot[start_index:end_index, :]
                ii = np.argmax(r_taue[:, 1])
                p_taue[m, 1] = r_taue[ii, 1]
                p_taue[m, 0] = r_taue[ii, 0]
            index_taue = np.where(p_taue[:, 0] != 0 )[0]
            rr = p_taue[index_taue,:]
            reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
            taue_04 = 1000 * (-10 - reg0[1]) / reg0[0]
            cor_04 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
            if cor_04.size == 1:
                cor_04 = np.array([[cor_04, 0.0]])
                
            #taue_05
            p_taue = np.zeros((10, 2))
            for m in range(1, 10):
                start_index = int((m) * 0.005 * fs )
                end_index = int((m+1) * 0.005 * fs )
                r_taue = lgacfplot[start_index:end_index, :]
                ii = np.argmax(r_taue[:, 1])
                p_taue[m, 1] = r_taue[ii, 1]
                p_taue[m, 0] = r_taue[ii, 0]
            index_taue = np.where(p_taue[:, 0] != 0 )[0]
            rr = p_taue[index_taue,:]
            reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
            taue_05 = 1000 * (-10 - reg0[1]) / reg0[0]
            cor_05 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
            if cor_05.size == 1:
                cor_05 = np.array([[cor_05, 0.0]])
    
            #taue_06
            p_taue = np.zeros((10, 2))
            for m in range(1, 10):
                start_index = int((m) * 0.006 * fs )
                end_index = int((m+1) * 0.006 * fs )
                r_taue = lgacfplot[start_index:end_index, :]
                ii = np.argmax(r_taue[:, 1])
                p_taue[m, 1] = r_taue[ii, 1]
                p_taue[m, 0] = r_taue[ii, 0]
            index_taue = np.where(p_taue[:, 0] != 0 )[0]
            rr = p_taue[index_taue,:]
            reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
            taue_06 = 1000 * (-10 - reg0[1]) / reg0[0]
            cor_06 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
            if cor_06.size == 1:
                cor_06 = np.array([[cor_06, 0.0]])            
    
            #taue_07
            p_taue = np.zeros((10, 2))
            for m in range(1, 10):
                start_index = int((m) * 0.007 * fs )
                end_index = int((m+1) * 0.007 * fs )
                r_taue = lgacfplot[start_index:end_index, :]
                ii = np.argmax(r_taue[:, 1])
                p_taue[m, 1] = r_taue[ii, 1]
                p_taue[m, 0] = r_taue[ii, 0]
            index_taue = np.where(p_taue[:, 0] != 0 )[0]
            rr = p_taue[index_taue,:]
            reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
            taue_07 = 1000 * (-10 - reg0[1]) / reg0[0]
            cor_07 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
            if cor_07.size == 1:
                cor_07 = np.array([[cor_07, 0.0]])
    
            #taue_08
            p_taue = np.zeros((10, 2))
            for m in range(1, 10):
                start_index = int((m) * 0.008 * fs )
                end_index = int((m+1) * 0.008 * fs )
                r_taue = lgacfplot[start_index:end_index, :]
                ii = np.argmax(r_taue[:, 1])
                p_taue[m, 1] = r_taue[ii, 1]
                p_taue[m, 0] = r_taue[ii, 0]
            index_taue = np.where(p_taue[:, 0] != 0 )[0]
            rr = p_taue[index_taue,:]
            reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
            taue_08 = 1000 * (-10 - reg0[1]) / reg0[0]
            cor_08 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
            if cor_08.size == 1:
                cor_08 = np.array([[cor_08, 0.0]])
                
            #taue_09
            p_taue = np.zeros((10, 2))
            for m in range(1, 10):
                start_index = int((m) * 0.009 * fs )
                end_index = int((m+1) * 0.009 * fs )
                r_taue = lgacfplot[start_index:end_index, :]
                ii = np.argmax(r_taue[:, 1])
                p_taue[m, 1] = r_taue[ii, 1]
                p_taue[m, 0] = r_taue[ii, 0]
            index_taue = np.where(p_taue[:, 0] != 0 )[0]
            rr = p_taue[index_taue,:]
            reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
            taue_09 = 1000 * (-10 - reg0[1]) / reg0[0]
            cor_09 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
            if cor_09.size == 1:
                cor_09 = np.array([[cor_09, 0.0]])
                
            #taue_10
            p_taue = np.zeros((10, 2))
            for m in range(1, 10):
                start_index = int((m) * 0.010 * fs )
                end_index = int((m+1) * 0.010 * fs )
                r_taue = lgacfplot[start_index:end_index, :]
                ii = np.argmax(r_taue[:, 1])
                p_taue[m, 1] = r_taue[ii, 1]
                p_taue[m, 0] = r_taue[ii, 0]
            index_taue = np.where(p_taue[:, 0] != 0 )[0]
            rr = p_taue[index_taue,:]
            reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
            taue_10 = 1000 * (-10 - reg0[1]) / reg0[0]
            cor_10 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
            if cor_10.size == 1:
                cor_10 = np.array([[cor_10, 0.0]])
                
            set_taue = np.array([[taue_03, taue_04, taue_05, taue_06, taue_07, taue_08, taue_09, taue_10],
                         [cor_03[0, 0], cor_04[0, 0], cor_05[0, 0], cor_06[0, 0], cor_07[0, 0], cor_08[0, 0], cor_09[0, 0], cor_10[0, 0]]]).T
    
            index_taue = np.argmax(set_taue[:, 1])
            corr = set_taue[index_taue, 1]
            taue = set_taue[index_taue, 0]
            Delta = (index_taue + 3) / 1000
        
            #taue_20
            if corr < 0.85:
                p_taue = np.zeros((10, 2))
                for m in range(1, 10):
                    start_index = int((m) * 0.020 * fs -1)
                    end_index = int((m+1) * 0.020 * fs -1)
                    r_taue = lgacfplot[start_index:end_index, :]
                    ii = np.argmax(r_taue[:, 1])
                    p_taue[m, 1] = r_taue[ii, 1]
                    p_taue[m, 0] = r_taue[ii, 0]
                index_taue = np.where(p_taue[:, 0] != 0 )[0]
                rr = p_taue[index_taue,:]
                reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
                taue_20 = 1000 * (-10 - reg0[1]) / reg0[0]
                cor_20 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
                if cor_20.size == 1:
                    cor_20 = np.array([[cor_20, 0.0]])
                if cor_20[0,0] > corr:
                    Delta = 0.02
                    corr = cor_20[0,0]
                    taue = taue_20
    
            #taue_30
            if corr < 0.85:
                p_taue = np.zeros((6, 2))
                for m in range(1, 6):
                    start_index = int((m) * 0.030 * fs -1)
                    end_index = int((m+1) * 0.030 * fs -1)
                    r_taue = lgacfplot[start_index:end_index, :]
                    ii = np.argmax(r_taue[:, 1])
                    p_taue[m, 1] = r_taue[ii, 1]
                    p_taue[m, 0] = r_taue[ii, 0]
                index_taue = np.where(p_taue[:, 0] != 0 )[0]
                rr = p_taue[index_taue,:]
                reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
                taue_30 = 1000 * (-10 - reg0[1]) / reg0[0]
                cor_30 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
                if cor_30.size == 1:
                    cor_30 = np.array([[cor_30, 0.0]])            
                if cor_30[0,0] > corr:
                    Delta = 0.03
                    corr = cor_30[0,0]
                    taue = taue_30
    
            #taue_40
            if corr < 0.85:
                p_taue = np.zeros((5, 2))
                for m in range(1, 5):
                    start_index = int((m) * 0.040 * fs -1)
                    end_index = int((m+1) * 0.040 * fs -1)
                    r_taue = lgacfplot[start_index:end_index, :]
                    ii = np.argmax(r_taue[:, 1])
                    p_taue[m, 1] = r_taue[ii, 1]
                    p_taue[m, 0] = r_taue[ii, 0]
                index_taue = np.where(p_taue[:, 0] != 0 )[0]
                rr = p_taue[index_taue,:]
                reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
                taue_40 = 1000 * (-10 - reg0[1]) / reg0[0]
                cor_40 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
                if cor_40.size == 1:
                    cor_40 = np.array([[cor_40, 0.0]])            
                if cor_40[0,0] > corr:
                    Delta = 0.04
                    corr = cor_40[0,0]
                    taue = taue_40
    
            #taue_50
            if corr < 0.85:
                p_taue = np.zeros((4, 2))
                for m in range(1, 4):
                    start_index = int((m) * 0.050 * fs -1)
                    end_index = int((m+1) * 0.050 * fs -1)
                    r_taue = lgacfplot[start_index:end_index, :]
                    ii = np.argmax(r_taue[:, 1])
                    p_taue[m, 1] = r_taue[ii, 1]
                    p_taue[m, 0] = r_taue[ii, 0]
                index_taue = np.where(p_taue[:, 0] != 0 )[0]
                rr = p_taue[index_taue,:]
                reg0 = np.polyfit(rr[:, 0], rr[:, 1], 1)
                taue_50 = 1000 * (-10 - reg0[1]) / reg0[0]
                cor_50 = -np.corrcoef(rr[:, 0], rr[:, 1])[0, 1]
                if cor_50.size == 1:
                    cor_50 = np.array([[cor_50, 0.0]])
                if cor_50[0,0] > corr:
                    Delta = 0.05
                    corr = cor_50[0,0]
                    taue = taue_50
                                                
            if taue < 0:
                for p in range(len(acfplot)):
                    if acfplot[p, 1] < 0.1:
                        taue = acfplot[p, 0]
                        break
                Delta = 0
                corr = 0
        
        # try:
        #     if Delta != 0:
        #         if Delta <= 0.01:
        #             taumax = 10 * Delta
        #         else:
        #             taumax = 0.2                                    
                            
        # except TypeError:
        #     taumax = None
        
        RunningFactor[z, :] = [z*runstep, Phi0, tau1*1000, phi1, 2 *width*1000, taue, 0.85]
        
    tau_e_array = RunningFactor[:, 5]
    tau_e_mean = np.mean(tau_e_array)
    tau_e_min = np.percentile(tau_e_array, 5)

    return tau_e_min, tau_e_mean#, RunningFactor

#%%
path = 'C:/Users/Fujitsu-A556/Documents/ING SON/IMA/Final/taue/Auralizaciones Finales'
savepath = path + '/Auralizacion_ACF.csv'
filelist_cruda = ff.batch_read(path, '.wav')#[2::14]

# QUITA LOS ARCHIVOS DESEADOS
filelist = filelist_cruda.copy()
# for file in filelist_cruda:
#     if ('KEMAR' in file) | ('ORGAN' in file) | ('LIRIC' in file) | ('JACSKON' in file) | ('BATA' in file) | ('CELLO' in file):
#         filelist.remove(file)

# Nomencla correctamente:
filenames = filelist.copy()
# for i, file in enumerate(filelist):
#     aux = file.split(' ')
#     if len(aux[1]) < 6:
#         file = aux[0] + '_0' + aux[1]
#     else:
#         file = aux[0] + '_' + aux[1]
#     filenames[i] = file
# filelist.sort()
# filenames.sort()
    
Nfiles = len(filelist)
largo = len(max(filenames, key=len)) - 4
largo = max(largo, len('Taue_mean'))
data = np.zeros((Nfiles+1, 3), dtype=f'<U{largo}')
data[0,:] = ['File', 'Taue_min', 'Taue_mean']

for i, file in enumerate(filelist):
    
    print(file)
    filepath = path + '/' + file
    try:
        int_val = 1.0
        taue_min, taue_mean = process(filepath, int_val)
    except:
        int_val = 2.0
        taue_min, taue_mean = process(filepath, int_val)
        print(int_val)
    data[i+1,:] = [filenames[i][:-4], f'{taue_min:.7f}', f'{taue_mean:.7f}']


np.savetxt(savepath, data, fmt='%s', delimiter=',')
    
print('ACF parameters have been saved.')
