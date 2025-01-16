import torch
import torch.nn as nn
from unet import Unet
from dataset import MembraneDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

# Define function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a U-Net model for segmentation')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

# Hyperparameters from command line arguments
learning_rate = 1e-3
batch_size = args.batch_size  # Get batch size from command line argument
epochs = args.epochs  # Get number of epochs from command line argument

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Dataset setup
dataset = MembraneDataset("data/membrane", "train")
dataset_size = len(dataset)
val_size = int(0.2 * dataset_size)  # 20% for validation
train_size = dataset_size - val_size

dataset_train, dataset_val = random_split(dataset, [train_size, val_size])

loader_train = DataLoader(dataset_train, batch_size=batch_size)
loader_val = DataLoader(dataset_val, batch_size=batch_size)

# Loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# Training loop
size = len(loader_train.dataset)
model.train()

# Outer loop for epochs
for epoch in range(epochs):
    running_loss = 0.0
    # Training phase
    with tqdm(loader_train, unit="batch", desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
        for batch, (image, label) in enumerate(tepoch):
            image = image.to(device)
            label = label.to(device)

            # Forward pass
            logits = model(image)
            pred_probab = nn.Softmax(dim=1)(logits)
            
            # Compute the loss
            loss = loss_fn(pred_probab, label.squeeze(1))

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate loss for the epoch
            running_loss += loss.item()

            # Print the loss periodically
            if batch % 100 == 0:
                loss_value = loss.item()
                current = batch * batch_size + len(image)
                tepoch.set_postfix(loss=loss_value, current=current, total=size)

    # Average loss for the epoch
    print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {running_loss / size:.4f}")

    # Validation phase (after each epoch)
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No need to compute gradients during validation
        for batch, (image, label) in enumerate(loader_val):
            image = image.to(device)
            label = label.to(device)

            # Forward pass
            logits = model(image)
            pred_probab = nn.Softmax(dim=1)(logits)

            # Compute the loss
            loss = loss_fn(pred_probab, label.squeeze(1))
            val_loss += loss.item()

    # Average validation loss
    print(f"Epoch [{epoch+1}/{epochs}] Validation Loss: {val_loss / len(loader_val):.4f}")

    # Set the model back to training mode
    model.train()
