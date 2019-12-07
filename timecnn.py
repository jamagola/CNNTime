####################
# Golam Gause Jaman#
# jamagola@isu.edu #
####################


# Generate experiment data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import scipy

rate=1000
window=100
slide=10
feature=1024
n=feature*slide+window-slide
group=3
p=100

u=[10,20,30]
s=[5,1,10]

labelsTrain=np.zeros(shape=(group*p))
labelsTest=np.zeros(shape=(group))

rawTrain=np.zeros(shape=(p*group,n))
rawTest=np.zeros(shape=(group,n))
rawFTrain=np.zeros(shape=(p*group,feature))
rawFTest=np.zeros(shape=(group,feature))

for i in np.arange(0,p):
    for j in np.arange(0,group):
        labelsTrain[i*group+j]=j
        rawTrain[i*group+j,:]=np.random.normal(u[j],s[j],n)
        
for j in np.arange(0,group):
    rawTest[j,:]=np.random.normal(u[j],s[j],n)
    labelsTest[j]=j

for i in np.arange(0,p):
    for j in np.arange(0,group):
        for k in np.arange(0,feature):
            #rawFTrain[i*group+j,k]=np.mean(rawTrain[i*group+j,k*slide:k*slide+window-1])
            rawFTrain[i*group+j,k]=np.std(rawTrain[i*group+j,k*slide:k*slide+window-1])
        
for j in np.arange(0,group):
    for k in np.arange(0,feature):
        #rawFTest[j,k]=np.mean(rawTest[j,k*slide:k*slide+window-1])
        rawFTest[j,k]=np.std(rawTest[j,k*slide:k*slide+window-1])
        
#Shuffle training raw data
index=np.arange(0,p*group)
np.random.shuffle(index)
rawFTrain=rawFTrain[index]
labelsTrain=labelsTrain[index]
# Shuffle done!
        
# Sample plot
plt.plot(rawTest[1,:], color='red')
plt.xlabel('Sample')
plt.ylabel('Readings')
plt.title('Sample raw sensor data')
plt.show()

plt.plot(rawFTest[1,:], color='green')
plt.xlabel('Sample')
plt.ylabel('Readings')
plt.title('Sample feature extracted data')
plt.show()

matFTrain=rawFTrain.reshape(p*group,32,32)
matFTest=rawFTest.reshape(group,32,32)

# Display as image
plt.contourf(matFTest[1,:,:])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Turning matrix to filled contour')
plt.show()

# Equivalent to torch.utils.data.DataLoader item
batch=5
channel=1
matFTest=torch.from_numpy(matFTest)
matFTest=matFTest.type(torch.FloatTensor)

matFTrain=torch.from_numpy(matFTrain)
matFTrain=matFTrain.type(torch.FloatTensor)

labelsTrain=torch.from_numpy(labelsTrain)
labelsTrain=labelsTrain.type(torch.LongTensor)

labelsTest=torch.from_numpy(labelsTest)
labelsTest=labelsTest.type(torch.LongTensor)

dataX=[] #TrainLoader
matFTrain=matFTrain.reshape(-1,batch,channel,32,32)
matFTest=matFTest.reshape(-1,1,channel,32,32)
labelsTrain=labelsTrain.reshape(-1,batch)

for i in np.arange(0,int(p*group/batch)):
    dataX.append((labelsTrain[i,:],matFTrain[i,:,:,:,:]))

# Output choices
classes = ('Red', 'Green', 'Blue')



#transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

import torch.nn as nn
import torch.nn.functional as F

# Defining network class
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,3)
        
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        
        return x

net=Net()


# Defining optimization / Loss function
import torch.optim as optim

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training
epo=100;

for epoch in range(epo):
    running_loss=0.0
    for i,data in enumerate(dataX,0):
        labels, inputs = data
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        if i % int(p*group/batch)==int(p*group/batch)-1:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/int(p*group/batch)))
            running_loss=0.0
print('Done!')

test=2
outputs=net(matFTest[test,:,:,:])
f=outputs.detach().numpy()
plt.plot(f[0])
plt.xlabel('Outputs')
plt.ylabel('Classification index')
plt.show()

temp, predicted = torch.max(outputs, 1)
print(temp)
print('Class predicted: {}'.format(classes[predicted]))
print('Class Tested: {}'.format(classes[labelsTest[test]]))