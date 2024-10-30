import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

# def blackman_tukey(signal, sample_freq, window='boxcar'):
#     # Calcular la correlación normalizada
#     count = len(signal)
#     lags = sig.correlation_lags(count, count)
#     valid = (np.abs(lags) < count)
#     lags = lags[valid]
    
#     unscaled_xcorr = sig.correlate(signal, signal, mode='full')[valid]
#     corr = unscaled_xcorr / (count - np.abs(lags))

#     # Resolución de la frecuencia
#     f_res = len(corr)
#     f_full = np.linspace(0, (f_res - 1) / f_res, f_res)
#     over_half = f_full > 0.5
#     f_half = f_full[~over_half]

#     # Representación en el dominio de la frecuencia
#     weighted_xcorr = corr * sig.windows.get_window(window, len(corr))
    
#     x_freq = np.fft.fft(weighted_xcorr)

#     # Calcular la PSD a partir de la representación en frecuencia    
#     psd = np.abs(x_freq)
#     psd = psd[~over_half]
#     psd[1:] *= 2  # Duplicar para obtener la energía total

#     return f_half, psd



# calcular la frecuencia de los latidos del corazon mirando a los maximos
# hay una variable del mat que indica cuando sucedes las ondas R del corazon, QRS detection
# filtrado espacial, es estadistico, se basa en la incorrelacion
# la idea es apilar realizaciones

def blackman_tukey(x,  M = None):    
    
    # N = len(x)
    x_z = x.shape
    
    N = np.max(x_z)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # hay que aplanar los arrays por np.correlate.
    # usaremos el modo same que simplifica el tratamiento
    # de la autocorr
    xx = x.ravel()[:r_len];

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    Px = Px.reshape(x_z)

    return Px;

def estimate_bandwidth(frequencies, psd):
    # Calcular la energía total
    total_energy = np.sum(psd)
    
    # Calcular el 99% de la energía total
    energy_threshold = 0.99 * total_energy
    
    # Acumular energía y encontrar límites
    cumulative_energy = np.cumsum(psd)
    
    # Encontrar las frecuencias que acumulan el 99% de la energía
    lower_limit_index = np.where(cumulative_energy >= (total_energy - energy_threshold))[0][0]
    upper_limit_index = np.where(cumulative_energy >= energy_threshold)[0][0]

    bandwidth = frequencies[upper_limit_index] - frequencies[lower_limit_index]
    
    # Calcular la frecuencia central
    central_frequency = (frequencies[lower_limit_index] + frequencies[upper_limit_index]) / 2
    
    return bandwidth, central_frequency, frequencies[lower_limit_index], frequencies[upper_limit_index]

def plot_signal_and_psd(signal, fs, title):
    N=len(signal)
    
    # Calcular la PSD usando el periodograma
    f_periodogram, Pxx_periodogram = sig.periodogram(signal, fs)
    
    # Calcular la PSD usando Welch
    nperseg = N//5 
    noverlap = nperseg // 2
    f_welch, Pxx_welch = sig.welch(signal, fs, nperseg=nperseg, noverlap=noverlap)

    # Calcular la PSD usando Blackman-Tukey
    # f_bt, Pxx_bt = blackman_tukey(signal, fs)
    
    # Estimar el ancho de banda usando Welch
    bandwidth_welch, central_freq_welch, lower_freq_welch, upper_freq_welch = estimate_bandwidth(f_welch, Pxx_welch)

    print('Ancho de banda '+title)
    print(bandwidth_welch)

    # Crear la figura y los ejes
    # fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.5)  # Espacio entre los subgráficos

    # Gráfico de la señal en el tiempo
    axs[0].plot(np.arange(len(signal)) / fs, signal)
    axs[0].set_title(f'Señal en el tiempo: {title}')
    axs[0].set_xlabel('Tiempo [s]')
    axs[0].set_ylabel('Amplitud')

    # Gráfico del periodograma
    axs[1].semilogy(f_periodogram, Pxx_periodogram)
    axs[1].set_title(f'PSD usando Periodograma: {title}')
    axs[1].set_xlabel('Frecuencia [Hz]')
    axs[1].set_ylabel('PSD [V**2/Hz]')

    # Gráfico de Welch
    axs[2].semilogy(f_welch, Pxx_welch)
    axs[2].set_title(f'PSD usando Welch: {title} (Ancho de banda: {bandwidth_welch:.2f} Hz)')
    axs[2].axvline(lower_freq_welch, color='r', linestyle='--', label=f'f1: {lower_freq_welch:.2f}')
    axs[2].axvline(central_freq_welch, color='b', linestyle=':', label=f'fc: {central_freq_welch:.2f}')
    axs[2].axvline(upper_freq_welch, color='g', linestyle='--', label=f'f2: {upper_freq_welch:.2f}')
    axs[2].legend()

    # Gráfico de Blackman-Tukey
    # axs[3].semilogy(f_bt, Pxx_bt)
    # axs[3].set_title(f'PSD usando Blackman-Tukey: {title}')
    # axs[3].set_xlabel('Frecuencia [Hz]')
    # axs[3].set_ylabel('PSD [V**2/Hz]')

    plt.tight_layout()
    plt.show()

##################
# Lectura de ECG #
##################

fs_ecg = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg = (mat_struct['ecg_lead']).flatten()
s_ecg = ecg[0:9999]

####################
# Lectura de pletismografía (PPG)  #
####################

fs_ppg = 400  # Hz
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

####################
# Lectura de audio #
####################

fs_audio, wav_data = sio.wavfile.read('silbido.wav')

# Asegurarse que wav_data sea unidimensional
if wav_data.ndim > 1:
    wav_data = wav_data[:, 0]  # Usar solo un canal si es estéreo

####################
# Graficar señales y PSDs #
####################

plot_signal_and_psd(s_ecg, fs_ecg, 'ECG')
plot_signal_and_psd(ppg, fs_ppg, 'PPG')
plot_signal_and_psd(wav_data, fs_audio, 'Audio')
