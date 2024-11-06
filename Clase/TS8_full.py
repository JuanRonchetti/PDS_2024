import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import welch

#%% Cargar el archivo .mat
data = scipy.io.loadmat('ECG_TP4.mat')

# Extraer las variables del archivo
ecg_lead = data['ecg_lead'].flatten()
qrs_detections = data['qrs_detections'].flatten()
heartbeat_pattern1 = data['heartbeat_pattern1'].flatten()
heartbeat_pattern2 = data['heartbeat_pattern2'].flatten()

# Parámetros
fs = 1000  # Frecuencia de muestreo en Hz
duration = 10  # Duración en segundos para la PSD

#%% Ancho de banda

# Calcular las distancias entre cada valor y el siguiente
distances = np.diff(qrs_detections) / fs

# Calcular la media y la varianza de las distancias
mean_distance = np.mean(distances)
er = np.std(distances)/mean_distance * 100

# Ancho de banda
bandwith = 1/mean_distance

# Mostrar resultados
print(f"Ancho de banda estimado: {bandwith:.2f} Hz")
print(f"Desvio/media: {er:.2f} %")

heart_rate_bpm = 60 / mean_distance

print(f"Frecuencia cardíaca estimada: {heart_rate_bpm:.2f} latidos por minuto")

#%% Determinar el pico máximo entre ambos latidos para normalizar
max_peak = max(np.max(np.abs(heartbeat_pattern1)), np.max(np.abs(heartbeat_pattern2)))

# Normalizar los latidos usando el pico máximo
heartbeat_pattern1_normalized = heartbeat_pattern1 / max_peak
heartbeat_pattern2_normalized = heartbeat_pattern2 / max_peak

# Alinear los latidos normal y ventricular
# Encontrar el índice del pico (R) en el latido normal
r_peak_index_normal = np.argmax(heartbeat_pattern1_normalized)

# Encontrar el índice del pico (R) en el latido ventricular
r_peak_index_ventricular = np.argmax(heartbeat_pattern2_normalized)

# Calcular la diferencia de índices
shift_amount = r_peak_index_normal - r_peak_index_ventricular

# Desplazar el latido ventricular para alinear con el latido normal
if shift_amount > 0:
    heartbeat_pattern2_aligned = np.pad(heartbeat_pattern2_normalized, (shift_amount, 0), 'constant')[:-shift_amount]
else:
    heartbeat_pattern2_aligned = heartbeat_pattern2_normalized[-shift_amount:]

# Graficar latidos normalizados y alineados
plt.figure(figsize=(12, 8))

#%% Subplot 1: Latido normal y ventricular alineados
plt.subplot(2, 1, 1)
time_axis_normal = np.arange(len(heartbeat_pattern1_normalized)) / fs * 1000  # Convertir a ms
time_axis_ventricular = np.arange(len(heartbeat_pattern2_aligned)) / fs * 1000  # Convertir a ms

plt.plot(time_axis_normal, heartbeat_pattern1_normalized, label='Latido Normal', color='green')
plt.plot(time_axis_ventricular, heartbeat_pattern2_aligned, label='Latido Ventricular', color='blue')
plt.title('Latidos Normal y Ventricular (Normalizados y Alineados)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud Normalizada')
plt.legend()
plt.grid()

#%% Subplot 2: Espectros normalizados de los latidos
plt.subplot(2, 1, 2)

N = len(heartbeat_pattern1)
nperseg = N // 5 
noverlap = nperseg // 2

# Welch para PSD
frequencies_heartbeat1, psd_heartbeat1 = welch(heartbeat_pattern1, fs=fs, nperseg=nperseg, noverlap=noverlap)
frequencies_heartbeat2, psd_heartbeat2 = welch(heartbeat_pattern2, fs=fs, nperseg=nperseg, noverlap=noverlap)

# Normalizo PSD entre 0 y 1
psd_heartbeat_min = min(np.min(np.abs(psd_heartbeat1)), np.min(np.abs(psd_heartbeat2)))
psd_heartbeat_max = max(np.max(np.abs(psd_heartbeat1)), np.max(np.abs(psd_heartbeat2)))

psd_heartbeat1_normalized = (psd_heartbeat1 - psd_heartbeat_min) / (psd_heartbeat_max - psd_heartbeat_min)
psd_heartbeat2_normalized = (psd_heartbeat2 - psd_heartbeat_min) / (psd_heartbeat_max - psd_heartbeat_min)

plt.semilogy(frequencies_heartbeat1, psd_heartbeat1_normalized, label='PSD Latido Normal', color='green')
plt.semilogy(frequencies_heartbeat2, psd_heartbeat2_normalized, label='PSD Latido Ventricular', color='blue')
plt.title('Densidad Espectral de Potencia (PSD) de los Latidos (Normalizadas)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral Normalizada (0-1)')
plt.legend()
plt.grid()

# Ajustar el layout y mostrar la figura
plt.tight_layout()
plt.show()

#%% Subplot de 3 filas para PSDs
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# PSD del ECG limpio (primeros 10 segundos)
ecg_clean_segment = ecg_lead[:duration * fs]
N = len(ecg_clean_segment)
nperseg = N // 5 
noverlap = nperseg // 2

frequencies_ecg_clean, psd_ecg_clean = welch(ecg_clean_segment, fs=fs, nperseg=nperseg, noverlap=noverlap)

axs[0].semilogy(frequencies_ecg_clean, psd_ecg_clean)
axs[0].set_title('PSD de los Primeros 10 Segundos de ECG Limpio')
axs[0].set_xlabel('Frecuencia (Hz)')
axs[0].set_ylabel('Densidad Espectral (V^2/Hz)')
axs[0].grid()

# Almacenar las PSDs de los latidos en ventanas específicas
psd_list = []
frequencies_list = []

for i in range(len(qrs_detections)):
    # Ventana alrededor del latido
    start_idx = int(qrs_detections[i] - (250 / 1000) * fs)
    end_idx = int(qrs_detections[i] + (350 / 1000) * fs)
    
    if start_idx >= 0 and end_idx < len(ecg_lead):
        heartbeat_segment = ecg_lead[start_idx:end_idx]
        
        N = len(heartbeat_segment)
        nperseg = N // 5 
        noverlap = nperseg // 2
        
        frequencies_beat, psd_beat = welch(heartbeat_segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        psd_list.append(psd_beat)
        frequencies_list.append(frequencies_beat)

        # Graficar la PSD individual como línea punteada
        axs[1].semilogy(frequencies_beat, psd_beat, linestyle='--', color='gray', alpha=0.5)

# Calcular la media de las PSDs normalizadas
mean_psd_normalized = np.mean(psd_list, axis=0)

# Graficar la media en línea continua
axs[1].semilogy(frequencies_list[0], mean_psd_normalized, label='Media PSD', color='black', linewidth=2)
axs[1].set_title('PSD de Latidos en Ventanas Específicas')
axs[1].set_xlabel('Frecuencia (Hz)')
axs[1].set_ylabel('Densidad Espectral Normalizada (0-1)')
axs[1].legend()
axs[1].grid()

# PSD de todo el registro
N = len(ecg_lead)
nperseg = N // 5 
noverlap = nperseg // 2

frequencies_full_ecg, psd_full_ecg = welch(ecg_lead, fs=fs, nperseg=nperseg, noverlap=noverlap)

axs[2].semilogy(frequencies_full_ecg, psd_full_ecg)
axs[2].set_title('PSD de Todo el Registro ECG')
axs[2].set_xlabel('Frecuencia (Hz)')
axs[2].set_ylabel('Densidad Espectral (V^2/Hz)')
axs[2].grid()

# Ajustar el layout y mostrar la figura final
plt.tight_layout()
plt.show()

#%% Nueva figura para comparar espectros
plt.figure(figsize=(12, 6))

# Graficar la diferencia entre el espectro completo y el espectro de los primeros 10 segundos
frequencies_full_ecg_diff = frequencies_full_ecg[:len(psd_ecg_clean)]
psd_diff = psd_ecg_clean - psd_full_ecg[:len(psd_ecg_clean)]

plt.plot(frequencies_full_ecg_diff, psd_diff)
plt.title('Diferencia entre el Espectro de los Primeros 10 Segundos y el Espectro Completo')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Diferencia en Densidad Espectral (V^2/Hz)')
plt.grid()

# Ajustar el layout y mostrar la figura final
plt.tight_layout()
plt.show()