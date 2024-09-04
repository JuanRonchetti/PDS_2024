import numpy as np
import matplotlib.pyplot as plt

# Parametros
fs = 1000
N = 1000 

ts = 1/fs

vmax = 1 
dc = 0 

df = 0.1
ff1 = 10 + df

noise = vmax/1

ph=0

# Implementacion
tt = np.linspace(0, (N-1)*ts, N)

# Eje de frecuencias
frecs = np.fft.fftfreq(N, d=1/fs)

## ff1    
aux_sin = vmax * np.sin( 2*np.pi*ff1*tt + ph ) + dc
aux_noise = np.random.uniform(-noise, noise, N)
aux = aux_sin + aux_noise
xx1 = aux.reshape(N,1)

XX1 = np.fft.fft(xx1, n=N, axis=0)
XX_abs1 = np.abs(XX1)
XX_angle1 = np.angle(XX1)

# ## ff2
# aux = vmax * np.sin( 2*np.pi*ff2*tt + ph ) + dc
# xx2 = aux.reshape(N,1)

# XX2 = np.fft.fft(xx2, n=N, axis=0)
# XX_abs2 = np.abs(XX2)
# XX_angle2 = np.angle(XX2)
        
# ## ff3
# aux = vmax * np.sin( 2*np.pi*ff3*tt + ph ) + dc
# xx3 = aux.reshape(N,1)

# XX3 = np.fft.fft(xx3, n=N, axis=0)
# XX_abs3 = np.abs(XX3)
# XX_angle3 = np.angle(XX3)

# Crear los subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=1) 

# Senial
axs[0].plot(tt, xx1, 'b', label = f'{ff1} Hz')
# axs[0].plot(tt, xx2, 'g', label = f'{ff2} Hz')
# axs[0].plot(tt, xx3, 'r', label = f'{ff3} Hz')
axs[0].set_title(f'Temporal (f: {ff1} Hz) (N: {N})')
axs[0].set_xlabel('Tiempo')
axs[0].set_ylabel('Amplitud')

# FFT Magnitud
axs[1].stem(frecs, XX_abs1, 'b' , label=f'{ff1} Hz')
# axs[1].stem(frecs, XX_abs2, 'g' , label=f'{ff2} Hz')
# axs[1].stem(frecs, XX_abs3, 'r' , label=f'{ff3} Hz')
axs[1].set_title('FFT Modulo')
axs[1].set_xlabel('Frecuencia')
axs[1].set_ylabel('Magnitud')

# FFT Fase
axs[2].stem(frecs, XX_angle1, 'b' , label=f'{ff1} Hz')
# axs[2].stem(frecs, XX_angle2, 'g' , label=f'{ff2} Hz')
# axs[2].stem(frecs, XX_angle3, 'r' , label=f'{ff3} Hz')
axs[2].set_title('FFT Fase')
axs[2].set_xlabel('Frecuencia')
axs[2].set_ylabel('Fase')
plt.legend()
plt.show()