# -------------------------------------------------------------------------------
#                                   Imports
# -------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# -------------------------------------------------------------------------------
#                                   Señales
# -------------------------------------------------------------------------------
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
 
f0 = fs / N  # frecuencia de senial normalizada
ts = 1/fs       # tiempo de muestreo
df = fs/N   # resolución espectral

frecs = np.fft.fftfreq(N, d=ts)

# SNR
SNR_db = 10 # db
SNR = 10**(SNR_db/10) # veces

a1 = 1

# Potencia de ruido
Ps = (a1**2)/2
Pr = Ps/SNR

# Frec
fr = np.random.uniform(-0.5, 0.5, N)
w0 = (N/4)
w1 = ( w0 + fr/N )

# Sampleo temporal
k = np.linspace(0, (N-1)*ts, N).flatten()

# Ruido
n = np.random.normal(0, np.sqrt(Pr), N)

# Armo la señal
signal = ( a1 * np.sin(2 * np.pi * w1 * k).flatten() ) + n
    
plt.plot(k, signal, label='Signal')
plt.title('Señal')
plt.xlabel('Tiempo [seg]')
plt.ylabel('Amplitud [V]')
plt.legend()

# Bucle
cant_realiz = 200
array_sig = np.zeros((cant_realiz, N))
array_fft = np.zeros((cant_realiz, N))
array_a1  = np.zeros(cant_realiz)

for i in range(cant_realiz):
    # Ruido
    n = np.random.normal(0, np.sqrt(Pr), N)
    
    # Frec
    fr = np.random.uniform(-0.5, 0.5, N)
    w1 = ( (N/4) + fr ) * df

    # Armo la señal
    signal = a1 * np.sin(2*np.pi*w1 * k).flatten() + n
    
    array_sig[i] = signal
    array_fft[i] = np.abs( np.fft.fft(signal)/N )
    array_a1[i]  = np.abs(np.fft.fft(signal)/N).item(int(N/4))
    
a1_est = np.mean(array_a1)


plt.figure(2)
plt.plot(frecs, 10*np.log10(array_fft[1]), label='FFT')
plt.title('Señal')
plt.xlabel('Frec')
plt.ylabel('Amplitud [V]')
plt.legend()