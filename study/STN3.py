from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STN3D(nn.Module):
    def __init__(self):
        super(STN3D, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1]).float())
        

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        print(f"conv1: {x.shape}")
        x = F.relu(self.bn2(self.conv2(x)))
        print(f"conv2: {x.shape}")
        x = F.relu(self.bn3(self.conv3(x)))
        print(f"conv3: {x.shape}")
        x = torch.max(x, 2, keepdim=True)[0]
        print(f"maxpool: {x.shape}")
        x = x.view(-1, 1024)
        print(f"view: {x.shape}")

        x = F.relu(self.bn4(self.fc1(x)))
        print(f"fc1: {x.shape}")
        x = F.relu(self.bn5(self.fc2(x)))
        print(f"fc2: {x.shape}")
        x = self.fc3(x)
        print(f"fc3: {x.shape}")

        x = x.view(-1, 3, 3)
        print(f"view: {x.shape}")
        return x
    
model = STN3D()
input_tensor = torch.randn(10, 3, 512)
print(f"input_tensor: {input_tensor.shape}")

print(f"#######################")
print("Start STN3D")
transform_matrix = model(input_tensor)
print(f"#######################")

input_tensor_for_matrix_multiplication = input_tensor.transpose(2, 1)
print(f"input_tensor_for_matrix_multiplication: {input_tensor_for_matrix_multiplication.shape}")

output = torch.bmm(input_tensor_for_matrix_multiplication, transform_matrix)
print(f"output: {output.shape}")

reshape_output = output.transpose(2, 1)
print(f"reshape_output: {reshape_output.shape}")