

import torch

n = 50
d1 = torch.ones(n)
d1 /= d1.sum()

d2 = torch.ones(n)
d2[0] = 100
d2 /= d2.sum()

d3 = torch.ones(n)
d3[0] = 1000
d3 /= d3.sum()

x = torch.rand(n)

for i, d in enumerate([d1, d2, d3]):
    a = (d * x).sum().log()
    b = (d * x.log()).sum()

    print((i, a, b, (a - b).abs()))
