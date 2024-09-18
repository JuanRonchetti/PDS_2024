#######################################################################################################################
#%% Configuración e inicio de la simulación
#######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# -------------------------------------------------------------------------------
#                                   Generador de señales
# -------------------------------------------------------------------------------

def my_signal_generator(ff, nn, fs, vmax=1, dc=0, ph=0):
    # Tiempo de muestreo
    ts = 1/fs
    
    # Sampleo temporal
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()

    aux = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
    xx = aux.reshape(nn, 1)
                        
    return (tt, xx)

# -------------------------------------------------------------------------------
#                                   Bloque cuantizador
# -------------------------------------------------------------------------------

def Quant(s_R, B, V_F):
    # --------------- Paso de cuantizacion
    q = V_F / (2 ** (B))
    
    # --------------- Señal cuantizada
    s_Q = np.round(s_R / q) * q
    
    return s_Q

# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras

# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 1
N_os = N * over_sampling
 
f0 = fs / N_os # frecuencia de senial normalizada

# Datos del ADC
B = 4 # bits
Vf = 2 # Volts
q = Vf / (2**B) # Volts
 
# datos del ruido
kn = 1
pot_ruido = (q**2 / 12) * kn # Watts (potencia de la señal 1 W)
 
ts = 1/fs # tiempo de muestreo
df = fs/N_os # resolución espectral
 
#######################################################################################################################
#%% Acá arranca la simulación

# Senoidal
tt_os, analog_sig = my_signal_generator(ff=f0, nn=N_os, fs=fs)
analog_sig = analog_sig.flatten()
tt = tt_os

B_values = [4, 8, 16]  # bits
kn_values = [0.1, 1, 10]  # relación de ruido
bins =10
for B in B_values:
    for kn in kn_values:
        # Recalcular parámetros según B y kn
        q = Vf / (2 ** B)  # Volts
        pot_ruido = (q**2 / 12) * kn  # Watts (potencia de la señal)

        # Ruido
        n = np.random.normal(0, np.sqrt(pot_ruido), len(tt_os))

        # Armo la señal
        sr = analog_sig + n
        tt = tt_os

        # Cuantizacion
        srq = Quant(sr, B, Vf)
        nq = sr - srq

        # Transformadas de Fourier
        ft_As = np.fft.fft(analog_sig, axis=0) / N_os  
        ft_SR = np.fft.fft(sr, axis=0) / N_os  
        ft_Srq = np.fft.fft(srq, axis=0) / N_os  
        ft_Nn = np.fft.fft(n, axis=0) / N_os  
        ft_Nq = np.fft.fft(nq, axis=0) / N_os  

        # Frecuencias
        ff = np.fft.fftfreq(N_os, d=1/fs)  
        bfrec = ff <= fs/2

        Nnq_mean = np.mean(np.abs(ft_Nq)**2)
        nNn_mean = np.mean(np.abs(ft_Nn)**2)

        ft_Srq_graph = 10 * np.log10(2 * np.abs(ft_Srq[bfrec])**2)
        ft_As_graph  = 10 * np.log10(2 * np.abs(ft_As[bfrec])**2)
        ft_Sr_graph  = 10 * np.log10(2 * np.abs(ft_SR[bfrec])**2)
        ft_Nn_graph  = 10 * np.log10(2 * np.abs(ft_Nn[bfrec])**2)
        ft_Nq_graph  = 10 * np.log10(2 * np.abs(ft_Nq[bfrec])**2)

        nNn_mean_graph = 10 * np.log10(2 * np.array([nNn_mean, nNn_mean]))
        Nnq_mean_graph = 10 * np.log10(2 * np.array([Nnq_mean, Nnq_mean]))

        # Presentación gráfica de los resultados en subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        # Primer gráfico: Señales en el tiempo
        axs[0].plot(tt, srq, lw=2, label='$ s_Q $ (ADC out)')
        axs[0].plot(tt, sr, linestyle=':', color='green', marker='o', markersize=3,
                    markerfacecolor='none', markeredgecolor='green', fillstyle='none',
                    label='$ s_R $ (ADC in)')
        axs[0].plot(tt_os, analog_sig, color='orange', ls='dotted', label='$ s $ (analog)')
        axs[0].set_title('Señal muestreada por un ADC de {:d} bits y kn={:.1f}'.format(B, kn))
        
        axs[0].set_xlabel('tiempo [segundos]')
        axs[0].set_ylabel('Amplitud [V]')
        axs[0].legend()

        # Segundo gráfico: Densidad de potencia en frecuencia
        axs[1].plot(ff[bfrec], ft_Srq_graph, lw=2, label='$ s_Q $ (ADC out)')
        axs[1].plot(ff[bfrec], ft_As_graph, color='orange', ls='dotted', label='$ s $ (analog)')
        axs[1].plot(ff[bfrec], ft_Sr_graph, ':g', label='$ s_R $ (ADC in)')
        
        axs[1].plot(ff[bfrec], ft_Nn_graph, ':r')
        axs[1].plot(ff[bfrec], ft_Nq_graph, ':c')
        
        axs[1].plot(np.array([ff[bfrec][0], ff[bfrec][-1]]), nNn_mean_graph,
                     '--r', label='$ \overline{n} =$' + '{:3.1f} dB'.format(10 * np.log10(2 * nNn_mean)))
        
        axs[1].plot(np.array([ff[bfrec][0], ff[bfrec][-1]]), Nnq_mean_graph,
                     '--c', label='$ \overline{n_Q} =$' + '{:3.1f} dB'.format(10 * np.log10(2 * Nnq_mean)))
        
        axs[1].set_title('Densidad de Potencia')
        axs[1].set_ylabel('Densidad de Potencia [dB]')
        axs[1].set_xlabel('Frecuencia [Hz]')
        axes_hdl_1 = axs[1].legend()
        axs[1].set_ylim((1.5*np.min(10*np.log10(2*np.array([Nnq_mean, nNn_mean]))), 10))
        axs[1].set_xlim(0, fs/2)

        # Tercer gráfico: Histograma del ruido cuantizado
        axs[2].hist(nq, bins=bins)
        axs[2].plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r')
        axs[2].set_title('Ruido de cuantización para {:d} bits y kn={:.1f}'.format(B, kn))
        axs[2].set_xlabel('Amplitud [V]')
        axs[2].set_ylabel('Frecuencia')

        plt.tight_layout()
        plt.show()