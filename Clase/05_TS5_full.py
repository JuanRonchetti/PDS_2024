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

# Frecuencias para SNRs
SNR_db_values = [3, 10]  # SNRs en dB
a1 = 1
w0 = N/4

# Histograma para almacenar resultados
a1_estimates = {snr: [] for snr in SNR_db_values}
omega_estimates = {snr: [] for snr in SNR_db_values}

# Variables para almacenar la última señal y su FFT
last_signals = {}
last_ffts = {}

for SNR_db in SNR_db_values:
    SNR = 10 ** (SNR_db / 10)  # Conversión a veces
    Ps = (a1 ** 2) / 2
    Pr = Ps / SNR
    
    for realization in range(cant_realiz):
        # Generar ruido
        n = np.random.normal(0, np.sqrt(Pr), N)
        
        # Frecuencia aleatoria
        fr = np.random.uniform(-0.5, 0.5)
        w1 = w0 + fr * df
        
        # Tiempo
        k = np.linspace(0, (N - 1) * ts, N).flatten()
        
        # Señal
        signal = a1 * np.sin(2 * np.pi * w1 * k) + n
        
        # FFT y estimación de a1
        fft_signal = np.abs(np.fft.fft(signal) / N)
        
        # Estimador en la frecuencia deseada (índice N/4)
        a1_estimate = fft_signal[int(N/4)] * 2  # Multiplicamos por 2 para corregir la amplitud
        
        # Almacenar estimaciones
        a1_estimates[SNR_db].append(a1_estimate)
        
        # Calcular el estimador de omega
        omega = np.argmax(fft_signal) * df  # Encuentra el índice máximo y lo convierte a frecuencia
        omega_estimate = np.abs( (N/2) - omega )
        omega_estimates[SNR_db].append(omega_estimate)
        
        # Guardar solo la última señal y su FFT para cada SNR
        if realization == cant_realiz - 1:
            last_signals[SNR_db] = signal
            last_ffts[SNR_db] = fft_signal

# Graficar señales y FFTs en subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
frecuencias = np.fft.fftfreq(N, d=ts)

# Señal para SNR de 3 dB
axs[0, 0].plot(np.linspace(0, (N-1)*ts, N), last_signals[3], label='Señal (SNR=3 dB)')
axs[0, 0].set_title('Señal en el Tiempo (SNR=3 dB)')
axs[0, 0].set_xlabel('Tiempo [s]')
axs[0, 0].set_ylabel('Amplitud [V]')
axs[0, 0].set_xlim(0, 100*ts) 
axs[0, 0].legend()
axs[0, 0].grid()

# FFT para SNR de 3 dB
axs[0, 1].plot(frecuencias[:N//2], last_ffts[3][:N//2], label='FFT (SNR=3 dB)')
axs[0, 1].set_title('FFT (SNR=3 dB)')
axs[0, 1].set_xlabel('Frecuencia [Hz]')
axs[0, 1].set_ylabel('Amplitud')
axs[0, 1].legend()
axs[0, 1].grid()

# Señal para SNR de 10 dB
axs[1, 0].plot(np.linspace(0, (N-1)*ts, N), last_signals[10], label='Señal (SNR=10 dB)', color='orange')
axs[1, 0].set_title('Señal en el Tiempo (SNR=10 dB)')
axs[1, 0].set_xlabel('Tiempo [s]')
axs[1, 0].set_ylabel('Amplitud [V]')
axs[1, 0].set_xlim(0, 100*ts)
axs[1, 0].legend()
axs[1, 0].grid()

# FFT para SNR de 10 dB
axs[1, 1].plot(frecuencias[:N//2], last_ffts[10][:N//2], label='FFT (SNR=10 dB)', color='orange')
axs[1, 1].set_title('FFT (SNR=10 dB)')
axs[1, 1].set_xlabel('Frecuencia [Hz]')
axs[1, 1].set_ylabel('Amplitud')
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()


# Graficar Histogramas en Subplots Horizontales
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Histograma de a_1
for snr in SNR_db_values:
    mean_a1_estimate = np.mean(a1_estimates[snr])  # Calcular la media de las estimaciones de a_1
    var_a1_estimate = np.var(a1_estimates[snr])     # Calcular la varianza de las estimaciones de a_1
    std_a1_estimate = np.std(a1_estimates[snr])     # Calcular el desvío estándar de las estimaciones de a_1
    percent_std_a1 = (std_a1_estimate / mean_a1_estimate) * 100  # Porcentaje del desvío estándar sobre la media

    axs[0].hist(a1_estimates[snr], bins=30, alpha=0.5, 
                label=f'SNR {snr} dB - $\mu$: {mean_a1_estimate:.2f}, $\sigma^2$: {var_a1_estimate:.2f}, $\sigma$: {std_a1_estimate:.2f} ({percent_std_a1:.2f}%)', 
                density=True)

axs[0].set_title('Histograma de $\\hat{a_1}$')
axs[0].set_xlabel('$\hat{a}_1$ [V]')
axs[0].set_ylabel('Densidad')
axs[0].axvline(a1, color='red', linestyle='--', label='Valor real $a_1$')
axs[0].legend()
axs[0].grid()
axs[0].set_xlim(a1*0.6, a1*1.4)  # Limitar el eje x para mejor visualización

# Histograma de Ω^1
for snr in SNR_db_values:
    mean_omega_estimate = np.mean(omega_estimates[snr])  # Calcular la media de las estimaciones de Ω^1
    var_omega_estimate = np.var(omega_estimates[snr])     # Calcular la varianza de las estimaciones de Ω^1
    std_omega_estimate = np.std(omega_estimates[snr])     # Calcular el desvío estándar de las estimaciones de Ω^1
    percent_std_omega = (std_omega_estimate / mean_omega_estimate) * 100  # Porcentaje del desvío estándar sobre la media

    axs[1].hist(omega_estimates[snr], bins=30, alpha=0.5, 
                label=f'SNR {snr} dB - $\mu$: {mean_omega_estimate:.2f}, $\sigma^2$: {var_omega_estimate:.2f}, $\sigma$: {std_omega_estimate:.2f} ({percent_std_omega:.2f}%)', 
                density=True)

axs[1].set_title('Histograma de $\\hat{\\Omega}_1$')
axs[1].set_xlabel('$\\hat{\\Omega}_1$ [Hz]')
axs[1].set_ylabel('Densidad')
axs[1].axvline(np.pi/2, color='red', linestyle='--', label='Valor real $\\Omega_0$')
axs[1].legend()
axs[1].grid()
axs[1].set_xlim(w0-1, w0+1)  # Limitar el eje x para mejor visualización

plt.tight_layout()
plt.show()