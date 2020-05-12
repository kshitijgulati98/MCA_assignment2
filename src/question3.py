#question 3

from os import listdir
import numpy as np
from scipy.io import wavfile
import librosa
import random
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.svm import LinearSVC

from question1 import spectrogram
from question2 import MFCC
from sklearn.metrics import precision_score,recall_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as prs

def noiseit(signal):
    noises = list()
    for i in listdir('./_background_noise_/'):
         fs,noise_sig = wavfile.read('./_background_noise_/' + i)
         noises.append([noise_sig,fs])
    noise_vec=random.choice(noises)
    vec=noise_vec[0]
    diff = len(vec) - len(signal)

    if diff > 0:
		# truncate
        signal=(signal + 0.0001*vec[0:len(signal)])
    elif diff < 0:
		# pad
        signal= (signal + 0.0001*np.concatenate(vec,np.zeros(-1*diff)))
    elif diff==0:
        signal= (signal + 0.0001*vec)

    return signal


def spec_features(file,nfft,overlap,fs,noise=False):
    signal, sr = librosa.load(file, sr=fs)
    if(noise  is True):
        signal=noiseit(signal)
    spec=spectrogram(signal,nfft,overlap,fs)

    
    return spec.flatten()
    
def mfcc_features(file,fs,nfft,overlap,coeff, noise=False):
    signal, sr = librosa.load(file, sr=fs)
    if(noise  is True):
        signal = noiseit(signal)
    mfcc= MFCC(signal,fs,nfft,overlap,coeff)
    return mfcc.flatten()

train_spec=[]
train_mfcc=[]
train_label=[]

val_spec =[]
val_mfcc = []
val_label = []


folders = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

nfft=2048
overlap=int(2048/1.2)
coeff=13
fs=16000


print("calculating training features...")
#training features
for idx , folder in enumerate(folders):
    train_path = './training/' + folder
    for f in listdir(train_path):
        filename = train_path + '/' + f
        train_spec.append(spec_features(filename,nfft,overlap,fs,noise=True))
        train_mfcc.append(mfcc_features(filename,fs,nfft,overlap,coeff,noise=True))
        train_label.append(idx)

print("calculated training features")  
print("calculating validation features....")      
        
#validation features
for idx , folder in enumerate(folders):
    train_path = './validation/' + folder
    for f in listdir(train_path):
        filename = train_path + '/' + f
        val_spec.append(spec_features(filename,nfft,overlap,fs))
        val_mfcc.append(mfcc_features(filename,fs,nfft,overlap,coeff))
        val_label.append(idx)

print("calculated validation features") 

dataset = dict()
dataset["train_spec"]=train_spec
dataset["train_mfcc"]=train_mfcc
dataset["train_label"]=train_label
dataset["val_spec"]=val_spec
dataset["val_mfcc"]=val_mfcc
dataset["val_label"]=val_label

#equalising the length

for idx, _ in enumerate(dataset["train_spec"]):
	array = dataset["train_spec"][idx]
	if array.shape[0] < 10250:
		dataset["train_spec"][idx] = np.concatenate((array,np.zeros(10250 - array.shape[0])))

for idx, _ in enumerate(dataset["val_spec"]):
	array = dataset["val_spec"][idx]
	if array.shape[0] < 1250:
		dataset["val_spec"][idx] = np.concatenate((array,np.zeros(10250- array.shape[0])))
        
for idx, _ in enumerate(dataset["train_mfcc"]):
	array = dataset["train_mfcc"][idx]
	if array.shape[0] < 640:
		dataset["train_mfcc"][idx] = np.concatenate((array,np.zeros(640 - array.shape[0])))

for idx, _ in enumerate(dataset["val_mfcc"]):
	array = dataset["val_mfcc"][idx]
	if array.shape[0] < 640:
		dataset["val_mfcc"][idx] = np.concatenate((array,np.zeros(640 - array.shape[0])))  
        
        
print('length equalisation completed')

classes_train = np.array(dataset['train_label'])
classes_test = np.array(dataset['val_label'])    

minmax=  MinMaxScaler(feature_range=(0,1),copy=True)

train_mfcc = (np.array(dataset['train_mfcc']))
test_mfcc = (np.array(dataset['val_mfcc']))

train_mfcc=minmax.fit_transform(train_mfcc)
test_mfcc=minmax.fit_transform(test_mfcc)

train_spec = normalize(np.array(dataset['train_spec']))
test_spec = normalize(np.array(dataset['val_spec']))

print('Model spectrogram is training')
model_spec = LinearSVC(random_state=0, tol=1e-5, max_iter = 1000, verbose = 1, C=0.05)
model_spec.fit(train_spec, classes_train)
print(model_spec.score(test_spec, classes_test))
Y=model_spec.predict(test_spec)
print(classification_report(Y,classes_test))

print('Model MFCC is training')
model_mfcc = LinearSVC(random_state=0, tol=1e-5, max_iter = 1000, C=0.05)
model_mfcc.fit(train_mfcc, classes_train)
print(model_spec.score(test_mfcc, classes_test))
Y_=model_mfcc.predict(test_mfcc)
print(classification_report(Y_,classes_test))
        





    
