import torch
import matplotlib.pyplot as plt

x = torch.load('outputs.pt')
print(len(x))
plt.hist(x)
plt.show()
