
import pickle
import matplotlib.pyplot as plt 

import numpy as np 
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
import random



with open(r"harmonics_feature_matrix", "rb") as fp:
       harmonics = pickle.load(fp) 


with open(r"tempogram_matrix", "rb") as fp:
       tempogram = pickle.load(fp)




#print(tempogram[0][0].shape)

train_fraction = 0.8
test_fraction = 0.2
harmonics_ = harmonics[3]
mapIndexPosition = list(zip(harmonics_, tempogram))
random.shuffle(mapIndexPosition)
harmonics_, tempogram = zip(*mapIndexPosition)

training_data = []
scaler = StandardScaler()
#random.shuffle(harmonics[3])

for i in range(int(train_fraction*len(harmonics_))):
      if harmonics_[i][0].shape[1] < 1293:
             harmonics_[i][0] = np.hstack((harmonics_[i][0], np.zeros(shape = (harmonics_[i][0].shape[0],1))))
      
      mel_fcc = harmonics_[i][0]
      label = harmonics_[i][1]

      
      for j in range(mel_fcc.shape[1]):
            if (j%128 == 0) & (j+128 < mel_fcc.shape[1] ):
                  
                  fcc_portion = mel_fcc[:, j: j+128]
                  scaler.fit(fcc_portion)
                  fcc_portion = scaler.transform(fcc_portion)
                  training_data.append([fcc_portion, label])
                  

testing_data = []
#print(len(training_data))

final_test_harmonics = []
for i in range(int(test_fraction*len(harmonics_))):
    
      i = i +1
      #print(i)
      #print(harmonics[3][-i][0].shape)

      mel_fcc =  harmonics_[-i][0]
      #print(mel_fcc.shape)
      label = harmonics_[-i][1]
      test_song = []
      
      for j in range(mel_fcc.shape[1]):
            if (j%128 == 0 )& (j+128 <= mel_fcc.shape[1] ) :
               
                  fcc_portion = mel_fcc[:, j: j+128]
                  scaler.fit(fcc_portion)
                  fcc_portion = scaler.transform(fcc_portion)
                  testing_data.append([fcc_portion, label])
                  test_song.append(fcc_portion)
      final_test_harmonics.append([test_song, label])
m = torch.nn.AvgPool2d(kernel_size = (3,1))




#print(testing_data[0][0].shape)
tempogram_new= []
for i in range(len(tempogram)):
    
     tempo =tempogram[i][0]
     label = tempogram[i][1]
   
     tempo= torch.unsqueeze(torch.Tensor(tempo), axis = 0 )
     #print(tempo.shape)
     tempo_averaged =torch.squeeze( m(tempo)).numpy()
     #print(tempo_averaged.shape)
    
     
     tempogram_new.append([tempo_averaged, label])
#print(len(tempogram_new))
#random.shuffle(tempogram_new)
training_data_temp = []

for i in range(int(train_fraction*len(tempogram_new))):
    
      
      mel_fcc =tempogram_new[i][0]
      #print(mel_fcc.shape)
      label = tempogram_new[i][1]
      
      
      for j in range(mel_fcc.shape[1]):
            if (j%128 == 0) & (j+128 < mel_fcc.shape[1] ) :
                  fcc_portion = mel_fcc[:, j: j+128]
                  scaler.fit(fcc_portion)
                  fcc_portion = scaler.transform(fcc_portion)
                  training_data_temp.append([fcc_portion, label])
testing_data_temp = []
final_test_tempogram= []
for i in range(int(test_fraction*len(tempogram_new))):
    
      i = i +1

      mel_fcc = tempogram_new[-i][0]
      label = tempogram_new[-i][1]
      test_song = []
      j_= []
      for j in range(mel_fcc.shape[1]):

            if (j%128 == 0) & (j+128 < mel_fcc.shape[1] ):
                  j_.append(j)
                  fcc_portion = mel_fcc[:, j: j+128]
                  scaler.fit(fcc_portion)
                  fcc_portion = scaler.transform(fcc_portion)
                  testing_data_temp.append([fcc_portion, label])

                  test_song.append(fcc_portion)
      print(len(j_))
      final_test_tempogram.append([test_song, label])

#print(len(testing_data_temp))



Train = []

for i in range(len(training_data)):
      ch1 = training_data[i][0]
      labels = training_data[i][1]
      #print(ch1.shape)
      ch2 = training_data_temp[j][0]
      dat = np.array([ch1,ch2])
      
      Train.append([dat , labels])


  


#print(len(Train))

Test = []

#print(len(Train))


for i in range(len(testing_data)):
      ch1 = testing_data[i][0]
      labels = testing_data[i][1]
      #print(ch1.shape)
      ch2 = testing_data_temp[i][0]
      dat = np.array([ch1,ch2])
      Test.append([dat ,labels])



Final_test = []

for i in range(len(final_test_tempogram)):
      har = final_test_harmonics[i][0]
      temp = final_test_tempogram[i][0]
      label = final_test_harmonics[i][1]
      song = []
      for j in range(len(har)):
           ch1 = har[j]
           ch2 = temp[j]
        
      
           dat = np.array([ch1,ch2])
           song.append(dat) 
      Final_test.append([song , label] )
'''
with open(r'Training_data', 'wb') as fp:
     pickle.dump(Train, fp)
with open(r'Testing_data', 'wb') as fp:
     pickle.dump(Test, fp)
with open(r'Final_Testing_data', 'wb') as fp:
     pickle.dump(Final_test, fp)
'''
