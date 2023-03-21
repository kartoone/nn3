import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Define linear model we will use below
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
	# modified from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
	# get our competing devices ready ... presumably cpu would somehow trigger the 16 core neural processors ... but I doubt it
	# mps uses the 48 core gpu instead
	mps_device = torch.device("mps")
	cpu_device = torch.device("cpu")

	# Download training data from open datasets.
	training_data = datasets.FashionMNIST(
	    root="data",
	    train=True,
	    download=True,
	    transform=ToTensor(),
	)
	
	# Download test data from open datasets.
	test_data = datasets.FashionMNIST(
	    root="data",
	    train=False,
	    download=True,
	    transform=ToTensor(),
	)

	# first hyperparam
	batch_size = 64
	
	# Create data loaders.
	train_dataloader = DataLoader(training_data, batch_size=batch_size)
	test_dataloader = DataLoader(test_data, batch_size=batch_size)
	
	# Show some sample data
	for X, y in test_dataloader:
	    print(f"Shape of X [N, C, H, W]: {X.shape}")
	    print(f"Shape of y: {y.shape} {y.dtype}")
	    break
	

	# sample code for working with m1
    # Create a Tensor directly on the mps device
    #x = torch.ones(5, device=mps_device)
    # Or
    #x = torch.ones(5, device="mps")
    # Any operation happens on the GPU
    #y = x * 2

    # Move your model to mps just like any other device
	model = NeuralNetwork().to(cpu_device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
	print("starting timer for training using CPU...")
	start = time.time()
	epochs = 30 
	for t in range(epochs):
	    print(f"Epoch {t+1}\n-------------------------------")
	    train(train_dataloader, model, loss_fn, optimizer, cpu_device)
	print(f"completed training in ... {time.time()-start}s")

	print("starting timer for testing using CPU...")
	start = time.time()
	test(test_dataloader, model, loss_fn, cpu_device)
	print(f"completed testing in ... {time.time()-start}s")
