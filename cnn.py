import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_epochs = 1000
batch_size = 4
lr = 1e-04

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5) , (0,5, 0.5, 0.5))])

train = torchvision.datasets.CelebA(root='./data', train=True,
download=True, transform=transform)
test = torchvision.datasets.CelebA(root='./data', train=False,
download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size,
                                        shuffle = True)
train_loader = torch.utils.data.DataLoader(test, batch_size = batch_size,
                                        shuffle = False)  

class CNN(nn.Module):
    def __init___(self):
        super(CNN, self ).__init__()
        self.conv = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fully_conn = nn.Linear(16*25, 120)
        self.fully_conn2 = nn.Linear(120, 84)
        self.fully_conn3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(f.relu(self.conv(x))) #flattening
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16*25) #flattened tensor
        x = f.relu(self.fully_conn(x))
        x = f.relu(self.fully_conn2(x))
        x = self.fully_conn3(x)
        return x
    
m = CNN().to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(m.parameters(), lr= lr)
n_steps = len(train_loader)

for i in range(n_epochs):
    for j, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        #fwdpass
        outputs = m(images)
        l = loss(outputs, labels)
        #bwckrd pass
        optimizer.zero_grad() #grad reset
        l.backward()
        optimizer.step()
        
        if (j+1) % 2000 == 0:
            print(f'Epoch [{i+1}/{n_epochs}')

        with torch.no_grad(): #do not need bckwrd pass
            correct = 0
            samples = 0
            correct_class = [0 for i in range(10)]
            class_samples = [0 for i in range(10)]
            for images, labels in test:
                images = images.to(device)
                labels = labels.to(device)
                outputs = m(images)

                _, pred = torch.max(outputs, 1)
                samples += labels.size(0)
                correct = (pred == labels).sum().item()

                for i in range(batch_size):
                    label = labels[i]
                    pred_i = pred[i]
                    if label == pred_i:
                        correct_class[label] += 1
                    class_samples[label] += 1

                
