import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io.wavfile import write

def vertical_flaten(a):
    return a.reshape(a.shape[0],1)

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')

s_ecg = (mat_struct['ecg_lead']).flatten()
N_ecg = len(s_ecg)

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

####################
# Lectura de audio #
####################

fs_audio, wav_data = sio.wavfile.read('silbido.wav')

####################
# Calculo de PSD   #
####################

fs = fs_ecg
N = N_ecg

nperseg = N//5
noverlap = nperseg//2

f, Pxx_den = sig.welch(s_ecg, fs, nperseg=nperseg, noverlap=noverlap)

plt.semilogy(f, Pxx_den)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()