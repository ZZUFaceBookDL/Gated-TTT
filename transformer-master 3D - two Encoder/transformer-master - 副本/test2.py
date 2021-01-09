import torch
import torch.nn.functional as F

loss_function = torch.nn.CrossEntropyLoss()

t1 = torch.ones(2, 3).long()
t2 = torch.zeros(2, 3).long()

reslut = loss_function(t1, t2)
print(reslut)