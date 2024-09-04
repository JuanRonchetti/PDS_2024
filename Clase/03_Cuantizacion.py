import numpy as np
import matplotlib.pyplot as plt

# Parámetros del ADC
n_bits = 4  # Número de bits del ADC
V_ref = 5.0  # Voltaje de referencia del ADC

# Calcular la resolución del ADC
n_levels = 2 ** n_bits
delta = V_ref / n_levels

# Generar una señal analógica continua (ejemplo: senoide)
t = np.linspace(0, 1, 1000)  # Tiempo
signal = V_ref/2 * (1 + np.sin(2 * np.pi * 5 * t))  # Señal senoide

# Cuantización de la señal
quantized_signal = np.round(signal / delta) * delta

# Graficar la señal original y la señal cuantizada
plt.figure(figsize=(12, 6))

# Señal original
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Señal Original')
plt.title('Señal Analógica Original')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()

# Señal cuantizada
plt.subplot(2, 1, 2)
plt.step(t, quantized_signal, where='post', label='Señal Cuantizada', color='r')
plt.title('Señal Cuantizada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()