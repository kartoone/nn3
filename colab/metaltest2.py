import torch
import time
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Define convolutional model
# Build the neural network, expand on top of nn.Module
# adapted from https://towardsdatascience.com/build-a-fashion-mnist-cnn-pytorch-style-efb297e22582
class ConvNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		
		# define layers
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
		
		self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
		self.fc2 = nn.Linear(in_features=120, out_features=60)
		self.out = nn.Linear(in_features=60, out_features=10)

	# define forward function
	def forward(self, t):
		# conv 1
		t = self.conv1(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)
		
		# conv 2
		t = self.conv2(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)
		
		# fc1
		t = t.reshape(-1, 12*4*4)
		t = self.fc1(t)
		t = F.relu(t)
		
		# fc2
		t = self.fc2(t)
		t = F.relu(t)
		
		# output
		t = self.out(t)
		# don't need softmax here since we'll use cross-entropy as activation.
		return t

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = F.cross_entropy(pred, y)

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
	model = ConvNetwork().to(cpu_device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	print("starting timer for training ConvNetwork using MPS...")
	start = time.time()
	epochs = 30 
	for t in range(epochs):
	    print(f"Epoch {t+1}\n-------------------------------")
	    train(train_dataloader, model, loss_fn, optimizer, cpu_device)
	print(f"completed training in ... {time.time()-start}s")

	print("starting timer for testing using MPS...")
	start = time.time()
	test(test_dataloader, model, loss_fn, cpu_device)
	print(f"completed testing in ... {time.time()-start}s")
