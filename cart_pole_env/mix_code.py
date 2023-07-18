import torch.nn as nn
import torch
import torch.nn.functional as F

class PP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(PP, self).__init__()
        self.l1 = nn.Linear(input_dim,128)
        self.l2 = nn.Linear(128,output_dim)
    def forward(self,x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return F.softmax(x,dim=1)

model = PP(4,2)
data = torch.rand(1,4)
print(model(data))