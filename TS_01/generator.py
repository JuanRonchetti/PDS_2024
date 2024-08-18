# -------------------------------------------------------------------------------
#                                   Imports
# -------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy.signal as sig
import matplotlib as mpl
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------
#                                   Funcion
# -------------------------------------------------------------------------------

def my_signal_generator( sig_type ):
    
    Ac   = sig_type['amplitud']
    Am   = sig_type['val_medio']
    frec = sig_type['frec']
    ph   = sig_type['fase']
    N    = sig_type['muestras']
    fs   = sig_type['frec_muestreo']
    
    # tiempo de muestreo
    ts = 1/fs
    
    # resolución espectral
    df = fs/N 
    
    # Sampleo temporal
    tt = np.linspace(0, (N-1)*ts, N).flatten()
    
    # Sampleo frecuencial
    ff = np.linspace(0, (N-1)*df, N).flatten()
    
    # estructuras de control de flujo
    if sig_type['tipo'] == 'senoidal':
    
        aux = Ac * np.sin( 2*np.pi*frec*tt + ph ) + Am
        x = aux.reshape(N,1)
    
    elif sig_type['tipo'] == 'cuadrada':
        
        aux = Ac * sig.square(2 * np.pi * frec * tt + ph) + Am
        x = aux.reshape(N,1)
        
    else:
        
        print("Tipo de señal no implementado.")        
        return
        
    # Presentación gráfica de los resultados
    
    plt.figure(1)
    line_hdls = plt.plot(tt, x)
    plt.title('Señal: ' + sig_type['tipo'] )
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [V]')
    #    plt.grid(which='both', axis='both')
    plt.show()