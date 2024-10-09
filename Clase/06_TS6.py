# -------------------------------------------------------------------------------
#                                   Imports
# -------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------
#                                   Parámetros
# -------------------------------------------------------------------------------

fs = 1000.0  # Frecuencia de muestreo (Hz)
N = 1000     # Cantidad de muestras
cant_realiz = 200  # Cantidad de realizaciones

# Frecuencia normalizada
f0 = fs / N  
ts = 1 / fs  
df = fs / N  

# Fijar SNR
SNR_db = 3  # SNR en dB
SNR = 10 ** (SNR_db / 10)  # Conversión a veces
a1 = 2
w0 = N/4
Ps = (a1 ** 2) / 2
Pr = Ps / SNR

# Histograma para almacenar resultados
a1_estimates = { 'Rectangular': [], 'Hann': [], 'Blackman': [] }
last_signals = {}
last_ffts = {}

# Ventanas a aplicar
ventanas = {
    'Rectangular': np.ones(N),
    'Hann':np.hanning(N),  
    'Blackman': np.blackman(N)  
}

for window_name, window in ventanas.items():
    for realization in range(cant_realiz):
        # Generar ruido
        n = np.random.normal(0, np.sqrt(Pr), N)
        
        # Frecuencia aleatoria
        fr = np.random.uniform(-2, 2)
        w1 = w0 + fr * df
        
        # Tiempo
        k = np.linspace(0, (N - 1) * ts, N).flatten()
        
        # Señal
        signal = a1 * np.sin(2 * np.pi * w1 * k) + n
        
        # Aplicar ventana a la señal
        windowed_signal = signal * window
        
        # FFT y estimación de a1
        fft_signal = np.abs(np.fft.fft(windowed_signal) / N)
        
        # Estimador en la frecuencia deseada (índice N/4)
        a1_estimate = fft_signal[int(w0)] * 2  # Multiplicamos por 2 para corregir la amplitud
        
        # Almacenar estimaciones
        a1_estimates[window_name].append(a1_estimate)
        
        # Guardar solo la última señal y su FFT para cada ventana
        if realization == cant_realiz - 1:
            last_signals[window_name] = windowed_signal  # Guardar la señal con ventana aplicada
            last_ffts[window_name] = fft_signal

# Graficar señales y FFTs en subplots
fig, axs = plt.subplots(3, 4, figsize=(16, 10))
frecuencias = np.fft.fftfreq(N, d=ts)

for i, (window_name, window) in enumerate(ventanas.items()):
    # Señal en el tiempo con ventana aplicada
    axs[i, 0].plot(np.linspace(0, (N-1)*ts, N), last_signals[window_name], label=f'Señal con {window_name}')
    axs[i, 0].set_title(f'Señal en el Tiempo ({window_name})')
    axs[i, 0].set_xlabel('Tiempo [s]')
    axs[i, 0].set_ylabel('Amplitud [V]')
    axs[i, 0].set_xlim(0, 1) 
    axs[i, 0].legend()
    axs[i, 0].grid()

    # FFT de la señal original (sin ventana)
    fft_original_signal = np.abs(np.fft.fft(signal) / N)
    axs[i, 1].plot(frecuencias[:N//2], 10*np.log10(fft_original_signal[:N//2]), label='FFT Señal Original')
    axs[i, 1].set_title(f'FFT Señal Original ({window_name})')
    axs[i, 1].set_xlabel('Frecuencia [Hz]')
    axs[i, 1].set_ylabel('Amplitud [dB]')
    axs[i, 1].legend()
    axs[i, 1].grid()

    # FFT de la ventana en sí misma
    fft_window = np.abs(np.fft.fft(window) / N)
    axs[i, 2].plot(frecuencias[:N//2], 10*np.log10(fft_window[:N//2]), label=f'FFT Ventana ({window_name})')
    axs[i, 2].set_title(f'FFT Ventana ({window_name})')
    axs[i, 2].set_xlabel('Frecuencia [Hz]')
    axs[i, 2].set_ylabel('Amplitud [dB]')
    axs[i, 2].legend()
    axs[i, 2].grid()
    
    # FFT para la señal con ventana aplicada
    axs[i, 3].plot(frecuencias[:N//2], 10*np.log10(last_ffts[window_name][:N//2]), label=f'FFT ({window_name})')
    axs[i, 3].set_title(f'FFT Señal*Ventana ({window_name})')
    axs[i, 3].set_xlabel('Frecuencia [Hz]')
    axs[i, 3].set_ylabel('Amplitud [dB]')
    axs[i, 3].legend()
    axs[i, 3].grid()

plt.tight_layout()
plt.show()

# Graficar Histograma de a_1 para las tres ventanas
plt.figure(figsize=(10,6))
for window_name in a1_estimates.keys():
    plt.hist(a1_estimates[window_name], bins=15, alpha=0.5,
             label=f'{window_name} - $\mu$: {np.mean(a1_estimates[window_name]):.2f}, $\sigma$: {np.std(a1_estimates[window_name]):.2f}', 
             density=True)

plt.title('Histograma de $\\hat{a_1}$ para diferentes ventanas')
plt.xlabel('$\hat{a}_1$ [V]')
plt.ylabel('Densidad')
plt.axvline(a1, color='red', linestyle='--', label='Valor real $a_1$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()