from model import Model
from scripts import train, evaluate_accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch

num_epochs=100

config={
    "conv1": {
        "in_channels": 1,
        "out_channels": 5,
        "kernel_size": 3,
        "pool_size": 2,
    },
    "conv2": {
        "in_channels": 5,
        "out_channels": 10,
        "kernel_size": 3,
        "pool_size": 2,
    },
    "flattened_size": 250,
    "dropout_rate": 0.45,
    "hidden_size": 128,
    "num_classes": 10,
}

def main():
    train_data=MNIST(root="./data", train=True, download=True, transform=ToTensor())
    train_loader=DataLoader(train_data, batch_size=64, shuffle=True)

    test_data=MNIST(root="./data", train=False, download=True, transform=ToTensor())
    test_loader=DataLoader(test_data, batch_size=64, shuffle=False)

    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=Model(config).to(device)
    loss_function=nn.CrossEntropyLoss()
    optimizer=Adam(model.parameters(), lr=0.001)

    train_acc_hist=[]
    test_acc_hist=[]

    for epoch in range(0, num_epochs):
        train(model, train_loader, loss_function, optimizer)
        acc=evaluate_accuracy(model, train_loader)
        train_acc_hist.append(acc)
        acc=evaluate_accuracy(model, test_loader)
        test_acc_hist.append(acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Train {train_acc_hist[-1]:.2%} | Test {acc:.2%}", flush=True)

    torch.save(model.state_dict(), f"{num_epochs}_epochs.pt")

    epochs=range(1, num_epochs+1)
    plt.plot(epochs, [100*acc for acc in train_acc_hist], label="Training")
    plt.plot(epochs, [100*acc for acc in test_acc_hist], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(epochs)
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
