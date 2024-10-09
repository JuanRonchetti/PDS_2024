# -------------------------------------------------------------------------------
#                                   Imports
# -------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd

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
Ps = (a1 ** 2) / 2
Pr = Ps / SNR

# Histograma para almacenar resultados
a1_estimates = { 'Rectangular': [], 'Hann': [], 'Blackman': [], 'Flat-Top': [] }
last_signals = {}
last_ffts = {}

# Ventanas a aplicar
ventanas = {
    'Rectangular':  sig.boxcar(N),
    'Hann':         sig.hann(N),
    'Blackman':     sig.blackman(N),
    'Flat-Top':     sig.flattop(N)
}

for window_name, window in ventanas.items():
    for realization in range(cant_realiz):
        # Generar ruido
        n = np.random.normal(0, np.sqrt(Pr), N)
        
        # Frecuencia aleatoria
        fr = np.random.uniform(-2, 2)
        w1 = N/4 + fr * df
        
        # Tiempo
        k = np.linspace(0, (N - 1) * ts, N).flatten()
        
        # Señal
        signal = a1 * np.sin(2 * np.pi * w1 * k) + n
        
        # Aplicar ventana a la señal
        windowed_signal = signal * ventanas[window_name]
        
        # FFT y estimación de a1
        fft_signal = np.abs(np.fft.fft(windowed_signal) / N)
        
        # Estimador en la frecuencia deseada (índice N/4)
        a1_estimate = ( fft_signal[int(N//4)] * 2 )
        
        # LA HANN Y LA BLACKMAN HAY QUE MULTIPLICAR POR 2 DE NUEVO, LA RECTANGULAR NO
        
        # Almacenar estimaciones
        a1_estimates[window_name].append(a1_estimate)
        
        # Guardar solo la última señal y su FFT para cada ventana
        if realization == cant_realiz - 1:
            last_signals[window_name] = windowed_signal  # Guardar la señal con ventana aplicada
            last_ffts[window_name] = fft_signal

# Graficar señales y FFTs en subplots para cada ventana
for window_name in ventanas.keys():
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    frecuencias = np.fft.fftfreq(N, d=ts)

    # Señal en el tiempo
    axs[0, 0].plot(np.linspace(0, (N-1)*ts, N), signal, label=f'Señal con {window_name}')
    axs[0, 0].set_title(f'Señal en el Tiempo ({window_name})')
    axs[0, 0].set_xlabel('Tiempo [s]')
    axs[0, 0].set_ylabel('Amplitud [V]')
    axs[0, 0].set_xlim(0, 1) 
    axs[0, 0].legend()
    axs[0, 0].grid()

    # FFT de la señal original (sin ventana)
    fft_original_signal = np.abs(np.fft.fft(signal) / N)
    axs[0, 1].plot(frecuencias[:N//2], 10*np.log10(fft_original_signal[:N//2]), label='FFT Señal Original')
    axs[0, 1].set_title(f'FFT Señal Original ({window_name})')
    axs[0, 1].set_xlabel('Frecuencia [Hz]')
    axs[0, 1].set_ylabel('Amplitud [dB]')
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Ventana en el tiempo
    axs[1, 0].plot(np.linspace(0, (N-1)*ts, N), ventanas[window_name], label=f'Ventana {window_name}')
    axs[1, 0].set_title(f'Ventana en el Tiempo ({window_name})')
    axs[1, 0].set_xlabel('Tiempo [s]')
    axs[1, 0].set_ylabel('Amplitud')
    axs[1, 0].legend()
    axs[1, 0].grid()

    # FFT de la ventana en sí misma
    window = ventanas[window_name]
    fft_window = np.abs(np.fft.fft(window, N) / N)
    axs[1, 1].plot(frecuencias, 10*np.log10(fft_window), label=f'FFT Ventana ({window_name})')
    axs[1, 1].set_title(f'FFT Ventana ({window_name})')
    axs[1, 1].set_xlabel('Frecuencia [Hz]')
    axs[1, 1].set_ylabel('Amplitud [dB]')
    axs[1, 1].legend()
    axs[1, 1].grid()
    
    # Señal multiplicada por la ventana en el tiempo
    axs[2, 0].plot(np.linspace(0, (N-1)*ts, N), last_signals[window_name])
    axs[2, 0].set_title(f'Señal*Ventana ({window_name})')
    axs[2, 0].set_xlabel('Tiempo [s]')
    axs[2, 0].set_ylabel('Amplitud')
    axs[2, 0].grid()

    # FFT para la señal con ventana aplicada
    axs[2, 1].plot(frecuencias[:N//2], 10*np.log10(last_ffts[window_name][:N//2]), label=f'FFT ({window_name})')
    axs[2, 1].set_title(f'FFT Señal*Ventana ({window_name})')
    axs[2, 1].set_xlabel('Frecuencia [Hz]')
    axs[2, 1].set_ylabel('Amplitud [dB]')
    axs[2, 1].legend()
    axs[2, 1].grid()
    
    plt.tight_layout()
    plt.show()

# Graficar Histograma de a_1 para las tres ventanas
resultados = []
plt.figure(figsize=(10,6))
for window_name in a1_estimates.keys():
    mean_a1_estimate = np.mean(a1_estimates[window_name])  # Media
    var_a1_estimate = np.var(a1_estimates[window_name])     # Varianza
    std_a1_estimate = np.std(a1_estimates[window_name])     # Desvio
    percent_std_a1 = (std_a1_estimate / mean_a1_estimate) * 100  # Desvio sobre la media
    sesgo_a1 = np.mean(np.array(a1_estimates[window_name]) - a1) # Sesgo
    resultados.append({
            'Ventana': window_name,
            'Media': mean_a1_estimate,
            'Sesgo': sesgo_a1,
            'Varianza': var_a1_estimate,
            'Desvio/Media': percent_std_a1
        })
    df_resultados = pd.DataFrame(resultados)
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