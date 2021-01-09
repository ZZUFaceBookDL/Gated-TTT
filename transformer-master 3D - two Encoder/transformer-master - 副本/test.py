import math

print(math.log(10000))  # 9.210340371976184

print(math.log(10000)/512)  # 0.017988946039015984

import torch

t1 = torch.Tensor([[1, 0],
                    [1, 1]])
print(t1)

t2 = t1.repeat(3, 3)
print(t2)

t3 = torch.Tensor(range(36)).reshape(6, 6)
print(t3)