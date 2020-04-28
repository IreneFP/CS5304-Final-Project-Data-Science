import numpy as np
import pandas as pd
from preprocess import processing
from sklearn.metrics import f1_score as f1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self, hiddenDim = 300):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300,hiddenDim)  # 6*6 from image dimension
        self.fc2 = nn.Linear(hiddenDim, 2)
        self.softmax =  nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def get_eval_data(self, data, mode = 'dev'):
        dataloader = torch.utils.data.DataLoader(data, batch_size = 1)
        y_stars = []
        if mode == 'dev':
            ys = [vec_targ[1] for vec_targ in data]       
            for i, data in enumerate(dataloader, 0):
                x, _ = data
                # print(x)
                output = self.forward(x).detach().numpy()[0]
                y_star = np.argmax(output)
                #print(y_star)
                y_stars.append(y_star)
        else:
            ys = []
            for i, data in enumerate(dataloader, 0):
                x = data
                output = self.forward(x).detach().numpy()[0]
                y_star = np.argmax(output)
                y_stars.append(y_star)

        return ys, y_stars

    def train(self, data, dev, verbose = True):
        trainloader = torch.utils.data.DataLoader(data, batch_size = 5000)
        criterion = nn.CrossEntropyLoss()
        # create your optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # print statistics
                running_loss += loss.item()
                if verbose and (i % 1 == 0): # print every 2000 mini-batches
                    ys, y_stars = self.get_eval_data(dev)
                    print('[%d, %5d] loss: %.3f\tDev FI: %.3f' % (epoch + 1, i + 1, running_loss, f1(ys, y_stars)))
                    running_loss = 0.0
        print('Finished Training.')

if __name__ == '__main__':
    print('Whoops, no main method built out.')