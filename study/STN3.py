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

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        iden2 = torch.Tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], requires_grad=True).repeat(batchsize, 1)
        print(f"iden2: {iden2.shape}, dtype: {iden2.dtype}")
        print(iden2)
        print(f"iden: {iden.shape}, dtype: {iden.dtype}")
        print(iden)
        print(f"x: {x}")
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        print(f"add: {x.shape}")
        x = x.view(-1, 3, 3)
        print(f"view: {x.shape}")
        return x
    
model = STN3D()
input_tensor = torch.randn(2, 3, 512)
print(f"input_tensor: {input_tensor.shape}")

output = model(input_tensor)
