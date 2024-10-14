
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import  MinMaxScaler
from collections import OrderedDict
from torch.autograd import Variable
import time

# neural network to predict the reward distributions for each state
class Net(nn.Module):
    def __init__(self ,input_shape ,output_shape):
        super(Net ,self).__init__()
        self.fc1 = nn.Linear(input_shape,100)
        self.fc2 = nn.Linear(100 ,100)
        self.fc3 = nn.Linear(100 ,output_shape)
    def forward(self ,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        return x