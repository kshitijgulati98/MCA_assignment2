#QUESTION 2
import math
import cmath
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fftpack import fft,dct

#MFCC  features
def MFCC(signal,fs,nfft,overlap,coeff,nfilt=30):
    #keep coeff=2-13
    #signal, fs= librosa.load(filename, mono=True,sr=fs, offset=0.0, duration=None)
    
    stft=np.zeros((int(len(signal) / (nfft-overlap)), int(nfft/2) + 1))
    
    windows  = np.arange(0,len(signal),nfft-overlap,dtype=int)
    windows=windows[windows + nfft < len(signal)]
    
    for i in range(len(windows)): 
        window=np.hamming(nfft)
        fr= signal[windows[i]:windows[i]+nfft]
        pre= fr* window
        fourier=fft(pre)
        spectrum=np.abs(np.square(fourier))/nfft
        stft[i,:] = spectrum[0:int(nfft/2) + 1]
        
    spec = np.array(stft)

    filterb=np.zeros((nfilt,int(np.floor((nfft/2)+1))))
    m = (2595 * np.log10(1 + (fs / 2) / 700))
    mel_points=[]
    for i in range(0,int(m),nfilt+2):
        mel_points.append(i)
    mel_points=np.array(mel_points)
    f=700 *(10**(mel_points/2595)-1)
    bins=[]
    for i in f:
        x=math.floor ((i/fs) * (nfft+1))
        bins.append(x)
    t=len(bins)    
    for i in range(1,nfilt+1):
        for j in range(int(bins[i-1]),int(bins[i])):
            filterb[i-1,j]=(j - bins[i-1])/(bins[i] - bins[i-1])
        
        for k in range(int(bins[i]),int(bins[i+1])):
            filterb[i-1,k] = (bins[i+1]-k)/(bins[i+1] - bins[i])
            
    fb=np.dot(spec,filterb.T)
    fb=20*np.log10(fb+0.00000000000000000000000000000000000001)
    ans = dct(fb, type=2, axis=1, norm='ortho')
    x,y=ans.shape
    mfcc= ans[0:x, 1 :(coeff+1)]

    
    return mfcc