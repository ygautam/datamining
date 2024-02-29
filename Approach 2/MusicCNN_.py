
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

with open(r'Training_data', 'rb') as fp:
    Train = pickle.load(fp)

with open(r'Testing_data', 'rb') as fp:
    Test = pickle.load(fp)


with open(r'Final_Testing_data', 'rb') as fp:
    Final_Test = pickle.load(fp)


__all__ = ['xception']

model_urls = {
    'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x



class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=10):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(2, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)



        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

import torch.utils.data as data_utils

device = 'cuda'
TRAIN_BATCH_SIZE =32
X_train = []
Y_train  = []
print(Train[0][1].shape)
for i in range(len(Train)):
    tr = torch.unsqueeze(torch.Tensor(Train[i][0]), axis = 0)
    X_train.append(tr)
    #print( Train[i][0].shape)
    lab = torch.unsqueeze(torch.Tensor(Train[i][1]), axis = 0)
    Y_train.append(lab)
    
x_train = torch.cat(X_train, axis = 0)
y_train = torch.cat(Y_train, axis = 0)

model = Xception().to(device)

model.load_state_dict(torch.load("./musicCNN2.bin"))
train_dataset = data_utils.TensorDataset(x_train , y_train)
train_dataloader = data_utils.DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE, shuffle = True
)

X_test = []
Y_test  = []
#print(Test[0][1].shape)
for i in range(len(Test)):
    te = torch.unsqueeze(torch.Tensor(Test[i][0]), axis = 0)
    X_test.append(te)
    #print( Train[i][0].shape)
    labte = torch.unsqueeze(torch.Tensor(Test[i][1]), axis = 0)
    #print(labte)
    Y_test.append(labte)
x_test = torch.cat(X_test, axis = 0)
y_test = torch.cat(Y_test, axis = 0)



test_dataset = data_utils.TensorDataset(x_test , y_test)
test_dataloader = data_utils.DataLoader(
    test_dataset,
    batch_size=TRAIN_BATCH_SIZE, shuffle = True
)
loss_criterion= torch.nn.CrossEntropyLoss()

lr =  2e-5
EPOCHS= 200
#TRAIN_BATCH_SIZE = 64

from sklearn import metrics
num_train_steps = int(len(train_dataset)/TRAIN_BATCH_SIZE * EPOCHS)
optimizer = AdamW(model.parameters(), lr = lr)
scheduler = get_linear_schedule_with_warmup(
                 optimizer , 
                 num_warmup_steps = 0,
                 num_training_steps= num_train_steps
                    )

acc= [0]
from sklearn.metrics import classification_report
'''
for epochs in range(EPOCHS): 
 model.train()
 print(f'\nTraining epoch = {epochs}\n')
 for bi, d in enumerate(train_dataloader):
    #if bi < 1:
       data  = d[0].to(device)
       label = d[1].to(device)
       #print(label.shape)
       class_ = torch.squeeze(torch.argmax(label, axis = -1))
       
    
       outs = model(data)

       loss = loss_criterion(outs, class_)
 
       loss.backward()
       optimizer.step()
       scheduler.step()
       if bi % 10 == 0 :
         print(f'bi = {bi}, loss = {loss}')
 model.eval()
 correct_ = 0
 total_ = 0
 outz = []
 targets = []
 print('\nValidating\n')
 for bi_, d_ in enumerate(test_dataloader):
   #if bi_ < 3 :
       correct = 0
       total = 0
       data  = d_[0].to(device)
       label = torch.squeeze(d_[1]).to(device)
       #print(label)
       outs_class = torch.squeeze(torch.argmax(label, axis = 1)).detach().cpu().numpy()
       targets.append(outs_class)
       outs = model(data)
       #print(outs_class)
      
       outs_class_ = torch.argmax(torch.nn.functional.softmax(outs), axis = 1).detach().cpu().numpy()
       outz.append(outs_class_)
       #print(outs_class_)
       for i in range(len(outs_class)):
           if outs_class[i] == outs_class_[i]:
                correct += 1
           total += 1 
      
       if bi_ % 10 == 0 :
             print(f'val_bi = {bi_}, val_acc_Score = {correct/total}')
 outz = np.concatenate(outz, axis = 0)
 targets = np.concatenate(targets , axis = 0)
 
 for i in range(len(outz)):
           if outz[i] == targets[i]:
                correct_ += 1
           total_ += 1
 
 print(f'\nEPoch, {epochs} \n acc + {correct_/total_}\n')  
 #print(classification_report(targets, outz, labels = np.arange(0,10)))
 acc.append(correct_/total_)
 if (correct_/total_) > max(acc[:-1]):
     torch.save(model.state_dict(), "./musicCNN2.bin")'''


def most_frequent(List): 
    return max(set(List), key = List.count) 
  
targs = []
pred = []
print(len(Final_Test[0]))
### Final Testing
model.eval()
for i in range(len(Final_Test)):
    data = Final_Test[i][0]


    labels = Final_Test[i][1]
    
    out_labels = []
    targs.append(np.argmax(labels))

    for j in data:
        
                x = torch.Tensor(j).to(device).unsqueeze(axis = 0)
                out = model(x)
                outs_class_ = torch.argmax(torch.nn.functional.softmax(out), axis = 1).detach().cpu().numpy()
             
                out_labels.append(outs_class_[0])
    
    pred.append(most_frequent(out_labels))
print(pred)
print(targs)
print(classification_report(targs, pred, labels = np.arange(0,10)))

