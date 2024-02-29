import librosa 
import os 
import numpy as np
import pickle



'''

labels = {}
cnt = 0
i= 0
j=9
data = []
for root,dirs,files in os.walk(r'./archive/genres/'):
      for filename in dirs:
           labels[filename] = np.array(([0]*i) + [1] + ([0]* j))
           i += 1 
           j -= 1
      for filename in files:
            label = filename[:filename.find('.')]
            y, sr = librosa.load('./archive/genres/'+ label + '/' +filename)
            
            data.append([y,labels[label]])
print(len(data))

print("Data recieved")


data_harmonic = []
data_percussion = [] 

cnt = 0 
for i in data:
    audio = i[0]
    
    harmonic_data = librosa.effects.harmonic(audio)
    
    data_harmonic.append([harmonic_data, i[1]])
       
    cnt += 1
    print(cnt)

print('Harmonic data recieved')
with open('data_harmonic', 'wb') as fp:
      pickle.dump(data_harmonic, fp)


cnt = 0 
for i in data:
    audio = i[0]
    
    
    percussion_data = librosa.effects.percussive(audio)
    data_percussion.append([harmonic_data, i[1]])
       
    cnt += 1
    print(cnt)



print('percussion data recieved')
 
with open('data_percussion', 'wb') as fp:
       pickle.dump(data_percussion, fp) 

print(data_harmonic[0][0].shape, data_percussive[0][0].shape,data[0][0].shape )

'''

with open('data_harmonic', 'rb') as fp:
       data_harmonic = pickle.load( fp)

#with open('data_percussion', 'rb') as fp:
 #      data_percussion = pickle.load( fp) 

print(len(data_harmonic))

features_matrix = []

cnt = 0
chroma_stft = []
chroma_cqt = []
chroma_cens = []
mel_spectrogram = []

mfcc = []

rms =[]

spectral_centroid =[]

spectral_bandwidth = []

spectral_contrast =[] 

spectral_flatness = []
spectral_rolloff = []


poly_features = []

tonnetz = []

zero_crossing_rate = []


cnt  = 0
for i in data_harmonic: 



    chroma_stft.append([librosa.feature.chroma_stft(i[0]),i[1]]) 

    chroma_cqt.append([librosa.feature.chroma_cqt(i[0]), i[1]])



    chroma_cens.append([librosa.feature.chroma_cens(i[0]), i[1]])




    mel_spectrogram.append([librosa.feature.melspectrogram(i[0]), i[1]])


    mfcc.append([librosa.feature.mfcc(i[0]), i[1]])




    rms.append([librosa.feature.rms(i[0]), i[1]])



    spectral_centroid.append([librosa.feature.spectral_centroid(i[0]), i[1]])



    spectral_bandwidth.append([librosa.feature.spectral_bandwidth(i[0]), i[1]])

    spectral_contrast.append([librosa.feature.spectral_contrast(i[0]), i[1]])




  

    spectral_flatness.append([librosa.feature.spectral_flatness(i[0]), i[1]])



    spectral_rolloff.append([librosa.feature.spectral_rolloff(i[0]) , i[1]])



    poly_features.append([librosa.feature.poly_features(i[0]), i[1]])




    tonnetz.append([librosa.feature.tonnetz(i[0]), i[1]])





    zero_crossing_rate.append([librosa.feature.zero_crossing_rate(i[0]) , i[1]])
    cnt += 1
    print(cnt)
feats_list = [chroma_stft, chroma_cqt, chroma_cens, mel_spectrogram, mfcc, rms, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff, poly_features, tonnetz, zero_crossing_rate ]

 
with open(r"harmonics_feature_matrix", "wb") as fp:
       pickle.dump(feats_list,fp) 


  






'''
cnt = 0

tempogram= []
fourier_tempogram = []
for i in data_percussion:
   tempogram.append([librosa.feature.tempogram(i[0]),  i[1]])
   fourier_tempogram.append([librosa.feature.fourier_tempogram(i[0]), i[1]])
   cnt += 1 
   print(cnt)




with open(r"tempogram_matrix", "wb") as fp:
       pickle.dump(tempogram,fp)

with open(r"Fourier_tempogram_matrix", "wb") as fp:
       pickle.dump(fourier_tempogram,fp)
'''
