import torch

batch = 100
n_samples = 64

x1 = torch.randn(batch, n_samples)
x2 = (torch.rand_like(x1) - 0.5) * 12**.5

m1 = x1.mean(dim=-1, keepdim=True)
m2 = x2.mean(dim=-1, keepdim=True)

cov = torch.mean((x1 - m1)*(x2 - m2), -1)

var1 = x1.var(dim=-1)
var2 = x2.var(dim=-1)

corr = cov / (var1 * var2) ** 0.5

y = x1 + x2
var_sum = y.var(dim=-1)

print(torch.mean(var_sum - var1 - var2))
print(torch.mean(corr))