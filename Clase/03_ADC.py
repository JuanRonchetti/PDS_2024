import numpy as np
import matplotlib.pyplot as plt

# Parametros sampling
fs = 1000
N = 1000 

ts = 1/fs

# Parametros senial
vmax = 1 
dc = 0
ph=0 

df = 0.1
ff1 = 10 + df

noise = vmax/4

# Parametros del ADC
n_bits = 4  
V_ref = 4

# LA VARIANZA NO HAY QUE FIJARLA SINO QUE HAY QUE FIJAR EL SNR ANALOGICO Y A PARTIR DE AHI OBTENER LA VARIANZA  

n_levels = 2 ** n_bits
delta = V_ref / n_levels

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

# ff1 cuantizada
quantized_xx1 = np.round(xx1 / delta) * delta

quantized_XX1 = np.fft.fft(quantized_xx1, n=N, axis=0)
quantized_XX_abs1 = np.abs(quantized_XX1)
quantized_XX_angle1 = np.angle(quantized_XX1)

# Crear los subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 8))
plt.subplots_adjust(hspace=1) 

# Señal Original
axs[0, 0].plot(tt, xx1, 'b', label=f'{ff1} Hz')
axs[0, 0].set_title(f'Señal Temporal Original (f: {ff1} Hz) (N: {N})')
axs[0, 0].set_xlabel('Tiempo')
axs[0, 0].set_ylabel('Amplitud')
axs[0, 0].grid(True)
axs[0, 0].legend()

# FFT Magnitud Original
axs[1, 0].stem(frecs, XX_abs1, 'b', label=f'{ff1} Hz')
axs[1, 0].set_title('FFT Magnitud Original')
axs[1, 0].set_xlabel('Frecuencia')
axs[1, 0].set_ylabel('Magnitud')
axs[1, 0].grid(True)
axs[1, 0].legend()

# FFT Fase Original
axs[2, 0].stem(frecs, XX_angle1, 'b', label=f'{ff1} Hz')
axs[2, 0].set_title('FFT Fase Original')
axs[2, 0].set_xlabel('Frecuencia')
axs[2, 0].set_ylabel('Fase')
axs[2, 0].grid(True)
axs[2, 0].legend()

# Señal Cuantizada
axs[0, 1].plot(tt, quantized_xx1, 'b', label=f'{ff1} Hz')
axs[0, 1].set_title(f'Señal Cuantizada (f: {ff1} Hz) (N: {N})')
axs[0, 1].set_xlabel('Tiempo')
axs[0, 1].set_ylabel('Amplitud')
axs[0, 1].grid(True)
axs[0, 1].legend()

# FFT Magnitud Cuantizada
axs[1, 1].stem(frecs, quantized_XX_abs1, 'b', label=f'{ff1} Hz')
axs[1, 1].set_title('FFT Magnitud Cuantizada')
axs[1, 1].set_xlabel('Frecuencia')
axs[1, 1].set_ylabel('Magnitud')
axs[1, 1].grid(True)
axs[1, 1].legend()

# FFT Fase Cuantizada
axs[2, 1].stem(frecs, quantized_XX_angle1, 'b', label=f'{ff1} Hz')
axs[2, 1].set_title('FFT Fase Cuantizada')
axs[2, 1].set_xlabel('Frecuencia')
axs[2, 1].set_ylabel('Fase')
axs[2, 1].grid(True)
axs[2, 1].legend()

plt.show()