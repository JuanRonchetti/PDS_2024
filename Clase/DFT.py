import numpy as np
import matplotlib.pyplot as plt

# Funcion
def my_DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        sum = 0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            sum += x[n] * (np.cos(angle) - 1j * np.sin(angle))
        X[k] = sum
    
    return X

# Parametros
fs = 8
ts = 1/fs
N = 8 
vmax = 1 
dc = 0 
ff = 1 
ph=0 

# Implementacion
tt = np.linspace(0, (N-1)*ts, N)

# Me gustaria saber bien que onda esta linea
frecs = np.fft.fftfreq(N, d=1) * (N/ff)
    
aux = vmax * np.sin( 2*np.pi*ff*tt + ph ) + dc
xx = aux.reshape(N,1)

XX = my_DFT(xx)
XX_abs = np.abs(XX)
        
# Crear los subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4) 

# Senial
axs[0].stem(tt, xx, use_line_collection=True)
axs[0].set_title('Senial en el dominio del tiempo')
axs[0].set_xlabel('Tiempo')
axs[0].set_ylabel('Amplitud')

# DFT
axs[1].stem(frecs, XX_abs , label='DFT')
axs[1].set_title('DFT')
axs[1].set_xlabel('Frecuencia')
axs[1].set_ylabel('Magnitud')