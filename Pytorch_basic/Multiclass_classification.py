#%%
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#%%
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

#%% Train_Ds and Test_DS
transform = transforms.ToTensor()
train_DS = datasets.MNIST(root = '/Users/sanghyun/Desktop/GIT_Folder', train=True, download=False, transform=transform) # transform -> tensor로 바꿔주는...!
test_DS = datasets.MNIST(root  = '/Users/sanghyun/Desktop/GIT_Folder', train=False, download=False, transform=transform)
