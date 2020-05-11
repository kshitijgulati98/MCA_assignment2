#QUESTION1

import math
import cmath
import numpy as np
import librosa
import matplotlib.pyplot as plt


#DISCRETE  FOURIER TRASNFORM
def DFT(signal,FFT_Size):

    for i in range(0,FFT_Size-len(signal)):
    	signal.append(0)
    ans = []
    for k in range(0,len(signal)):
    	sumR = 0;
    	sumI = 0;
    	for i in range(0,len(signal)):
    		theta = (2*math.pi*k*i)/len(signal)
    		sumR+=cmath.cos(theta)*signal[i]
    		sumI+=cmath.sin(theta)*signal[i]
    	ans.append((sumR**2 + sumI**2))
    return np.array(ans)

#SPECTROGRAM
#returns  spectrogram array and displays  plot.
#NFFT-window  size
def spectrogram(signal, nfft,  overlap,fs):
    #signal, sampling_rate= librosa.load(filename, mono=True,sr=fs, offset=0.0, duration=None)
    
    stft=np.zeros((int(len(signal) / (nfft-overlap)), int(nfft/2) + 1))
    
    windows  = np.arange(0,len(signal),nfft-overlap,dtype=int)
    windows=windows[windows + nfft < len(signal)]
    
    for i in range(len(windows)): 
        window=np.hanning(nfft)
        fr= signal[windows[i]:windows[i]+nfft]
        pre= fr* window
        fourier=DFT(pre,nfft)
        spectrum=np.abs(np.square(fourier))/nfft
        stft[i,:] = spectrum[0:int(nfft/2) + 1]
        
    spec = np.array(stft)
    finx = 20*np.log10(abs(spec))
    plt.imshow(finx.T, origin="lower", aspect="auto", cmap="inferno", interpolation="none")
    plt.colorbar()
    plt.show()
    
    return(finx.T)
