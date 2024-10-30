import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

def blackman_tukey(x, fs, M=None):    
    x_z = x.shape
    N = np.max(x_z)
    
    if M is None:
        M = N // 5
    
    r_len = 2 * M - 1
    xx = x.ravel()[:r_len]
    
    r = np.correlate(xx, xx, mode='same') / r_len
    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n=N))
    
    freq = np.fft.fftfreq(N, 1/fs)
    
    return freq, Px

def estimate_bandwidth(frequencies, psd):
    total_energy = np.sum(psd)
    energy_threshold = 0.995 * total_energy
    
    cumulative_energy = np.cumsum(psd)
    
    lower_limit_index = np.where(cumulative_energy >= (total_energy - energy_threshold))[0][0]
    upper_limit_index = np.where(cumulative_energy >= energy_threshold)[0][0]

    bandwidth = frequencies[upper_limit_index] - frequencies[lower_limit_index]
    central_frequency = (frequencies[lower_limit_index] + frequencies[upper_limit_index]) / 2
    
    return bandwidth, central_frequency, frequencies[lower_limit_index], frequencies[upper_limit_index]

def plot_signal_and_psd(signal, fs, title):
    N = len(signal)
    
    # Calcular la PSD usando diferentes métodos
    f_periodogram, Pxx_periodogram = sig.periodogram(signal, fs)
    nperseg = N // 5 
    noverlap = nperseg // 2
    f_welch, Pxx_welch = sig.welch(signal, fs, nperseg=nperseg, noverlap=noverlap)
    f_bt, Pxx_bt = blackman_tukey(signal, fs)

    # Filtrar frecuencias positivas para Blackman-Tukey
    bfrec_bt = f_bt >= 0

    # Estimar el ancho de banda usando Welch
    bandwidth_welch, central_freq_welch, lower_freq_welch, upper_freq_welch = estimate_bandwidth(f_welch, Pxx_welch)

    print(f'Ancho de banda {title}: {bandwidth_welch:.2f} Hz')

    # Obtener límites Y para unificar
    all_psd_values = np.concatenate([Pxx_periodogram, Pxx_welch, Pxx_bt[bfrec_bt]])
    
    # Evitar valores cero para el logaritmo
    all_psd_values += 1e-10  # Añadir un pequeño valor para evitar log(0)
    
    y_min, y_max = np.min(all_psd_values), np.max(all_psd_values)

    # Crear la figura y los ejes
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.5)  # Espacio entre los subgráficos

    # Gráfico de la señal en el tiempo
    axs[0].plot(np.arange(len(signal)) / fs, signal)
    axs[0].set_title(f'Señal en el tiempo: {title}')
    axs[0].set_xlabel('Tiempo [s]')
    axs[0].set_ylabel('Amplitud')

    # Gráfico del periodograma
    axs[1].semilogy(f_periodogram, Pxx_periodogram + 1e-10)  # Evitar log(0)
    axs[1].set_title(f'PSD usando Periodograma: {title}')
    axs[1].set_xlabel('Frecuencia [Hz]')
    axs[1].set_ylabel('PSD [V**2/Hz]')
    
    # Establecer límites Y iguales para el periodograma
    axs[1].set_ylim(y_min, y_max)

    # Gráfico de Welch
    axs[2].semilogy(f_welch, Pxx_welch + 1e-10)  # Evitar log(0)
    axs[2].set_title(f'PSD usando Welch: {title} (Ancho de banda: {bandwidth_welch:.2f} Hz)')
    
    # Líneas para los límites del ancho de banda y la frecuencia central
    axs[2].axvline(lower_freq_welch, color='r', linestyle='--', label='Límite inferior')
    axs[2].axvline(upper_freq_welch, color='g', linestyle='--', label='Límite superior')
    axs[2].axvline(central_freq_welch, color='b', linestyle=':', label='Frecuencia central')
    
    axs[2].legend()
    axs[2].set_ylim(y_min, y_max)
    
    # Gráfico de Blackman-Tukey
    axs[3].semilogy(f_bt[bfrec_bt], Pxx_bt[bfrec_bt] + 1e-10)  # Evitar log(0)
    axs[3].set_title(f'PSD usando Blackman-Tukey: {title}')
    axs[3].set_xlabel('Frecuencia [Hz]')
    axs[3].set_ylabel('PSD [V**2/Hz]')
    
    # Establecer límites Y iguales para Blackman-Tukey
    axs[3].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()

##################
# Lectura de ECG #
##################
fs_ecg = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg = (mat_struct['ecg_lead']).flatten()
s_ecg = ecg[:9999]

####################
# Lectura de pletismografía (PPG)  #
####################
fs_ppg = 400  # Hz
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir cabecera si existe

####################
# Lectura de audio #
####################
fs_audio, wav_data = sio.wavfile.read('silbido.wav')
if wav_data.ndim > 1:
    wav_data = wav_data[:, 0]  # Usar solo un canal si es estéreo

####################
# Graficar señales y PSDs #
####################
plot_signal_and_psd(s_ecg, fs_ecg, 'ECG')
plot_signal_and_psd(ppg, fs_ppg, 'PPG')
plot_signal_and_psd(wav_data, fs_audio, 'Audio')