#%%
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import  optim
import matplotlib.pyplot as plt

#%%
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

#%% Train_Ds and Test_DS
transform = transforms.ToTensor()
train_DS = datasets.MNIST(root = '/Users/sanghyun/Desktop/GIT_Folder', train=True, download=False, transform=transform) # transform -> tensor로 바꿔주는...!
test_DS = datasets.MNIST(root  = '/Users/sanghyun/Desktop/GIT_Folder', train=False, download=False, transform=transform)

#%% DatLoader
BATCH_SIZE = 32
train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True)

#%% Model

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(28 * 28, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 10))
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

#%% train & test function
def Train(model, train_DL, criterion, optimizer, EPOCH, LP):
    loss_history = []
    NoT = len(train_DL.dataset)
    model.train()

    for ep in range(EPOCH):

        rloss = 0

        for x_batch, y_batch in train_DL:

            # inference
            y_hat = model(x_batch)

            # loss
            loss = criterion(y_hat, y_batch)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss accumulation
            loss_b = loss.item() * x_batch.shape[0]
            rloss += loss_b

        loss_e = rloss / NoT
        loss_history += [loss_e]
        print(f'Epoch : {ep + 1}, train loss : {round(loss_e, 3)}')
        print("-" * 20)

    return  loss_history


def Test(model, test_DL):
    model.eval()

    with torch.no_grad():
        rcorrect = 0
        for x_batch, y_batch in test_DL:
            y_hat = model(x_batch)
            pred = y_hat.argmax(dim = 1)
            correct_b = torch.sum(pred == y_batch).item()
            rcorrect += correct_b
        accuracy_e = rcorrect / len(test_DL.dataset) * 100

    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(accuracy_e, 1)} %)")

#%% Train

if __name__ == "__main__" :
    model = MLP()
    LP = 1e-3
    EPOCH = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LP)

    loss_history = Train(model, train_DL, criterion, optimizer, EPOCH, LP)

    Test(model, test_DL)


