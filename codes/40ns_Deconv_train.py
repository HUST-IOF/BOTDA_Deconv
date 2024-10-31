
import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
from torch import nn, optim
import torch.optim.lr_scheduler
from scipy import io
import copy
import time
from model2 import resnet34

# Select device, prioritize using GPU (CUDA); if unavailable, fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Custom dataset class for loading feature and label data
class TestDataset(data.Dataset):
    def __init__(self, lab_dir, fea_dir, transform=None):
        # Initialize data paths and load data
        self.lab_dir = lab_dir  # Path to label data
        self.fea_dir = fea_dir  # Path to feature data
        self.fea_list = np.load(fea_dir, allow_pickle=True)  # Load feature data
        self.lab_list = np.load(lab_dir, allow_pickle=True)  # Load label data

    def __getitem__(self, idx):
        # Get feature and label for the given index
        feature = self.fea_list[:, :, idx]  # Extract feature data
        feature = feature.squeeze()  # Remove single-dimensional entries
        feature = torch.FloatTensor(feature)  # Convert to PyTorch tensor
        label = self.lab_list[idx, 20:520]  # Extract label data from columns 20 to 520
        label = torch.FloatTensor(label)  # Convert to PyTorch tensor
        return feature, label  # Return feature and label

    def __len__(self):
        # Return the length of the dataset (i.e., number of samples)
        dataset_len = int(np.size(self.lab_list, 0))
        return dataset_len



# Main function to train and evaluate the model
if __name__ == "__main__":
    save_path = './models/resnet34_5point_40ns_71.pth'  # Path to save the model
    model = resnet34().to(device)  # Create and move model to GPU

    # Initialize parameters for tracking the best model
    bestNet = copy.deepcopy(model.state_dict())  # Deep copy of model parameters
    bestEpoch = 0  # Track the best epoch
    bestLoss = 1.0  # Track the best loss value
    startTime = time.time()  # Record the start time of training
    wait = 0  # Early stopping counter

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer with learning rate 1e-4

    lab_dir = f'./dataset\\40ns_train\\simulation_data_71points_BFS.npy'
    fea_dir = f'./dataset\\40ns_train\\simulation_data_71points_BGS.npy'
    test_lab_dir = f'./dataset\\40ns_test\\simulation_data_71points_BFS.npy'
    test_fea_dir = f'./dataset\\40ns_test\\simulation_data_71points_BGS.npy'
    val_lab_dir = f'./dataset\\40ns_val\\simulation_data_71points_BFS.npy'
    val_fea_dir = f'./dataset\\40ns_val\\simulation_data_71points_BGS.npy'

    # Load training, validation, and testing datasets
    train_dataset = TestDataset(lab_dir, fea_dir)
    val_dataset = TestDataset(val_lab_dir, val_fea_dir)
    test_dataset = TestDataset(test_lab_dir, test_fea_dir)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=12, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False)

    # Training process parameters
    minEpoch = 100  # Minimum training epochs
    maxEpoch = 400  # Maximum training epochs
    maxWait = 150  # Early stopping patience

    trainLossPlot = []  # To record training loss
    valLossPlot = []  # To record validation loss

    # Start the training loop
    n_epoch = maxEpoch  # Set maximum number of epochs
    for epoch in range(n_epoch):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Cumulative loss for each epoch
        trainLoss = 0.0  # Total training loss
        print(f'Epoch: {epoch + 1}')
        for j, data in enumerate(train_loader, 0):
            feature, label = data  # Get batch data
            feature, label = feature.to(device), label.to(device)  # Move data to device (GPU/CPU)
            optimizer.zero_grad()  # Clear the gradients
            out = model(feature.unsqueeze(1))  # Forward pass, compute model output
            label = label.squeeze()  # Squeeze label dimensions
            out = torch.squeeze(out)[:, 20:520]   # Squeeze model output
            loss = nn.MSELoss()(out, label)  # Calculate mean squared error loss
            loss.backward()  # Backward pass, compute gradients
            optimizer.step()  # Update parameters
            trainLoss += loss.item()  # Accumulate training loss

            # Print and log loss
            if j % 500 == 0 and j != 0:
                print('[%d, %5d] loss: %.12f' % (epoch + 1, j, running_loss / 500))
                running_loss = 0.0

            # Calculate average training loss
        trainLoss = trainLoss / (len(train_loader))
        print('train_loss = ', trainLoss)
        trainLossPlot.append(trainLoss)

        # Validate the model
        valLoss = 0.0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            for data in val_loader:
                val_feature, val_label = data  # Get validation data
                val_feature, val_label = val_feature.to(device), val_label.to(device)
                predicted = model(val_feature.unsqueeze(1))  # Make predictions
                val_label = val_label.squeeze()
                predicted = torch.squeeze(predicted)[:, 20:520]  # Process model output
                loss = nn.MSELoss()(predicted, val_label)  # Calculate validation loss
                valLoss += loss.item()
            valLoss = valLoss / len(val_loader)
            print('val_loss = ', valLoss)
            valLossPlot.append(valLoss)

        # Save the best model
        if bestLoss > valLoss:
            bestLoss = valLoss  # Update the best loss
            bestNet = copy.deepcopy(model.state_dict())  # Save the best model parameters
            bestEpoch = epoch + 1  # Record the best epoch
            state = {
                'state_dict': model.state_dict(),
                'bestEpoch': bestEpoch,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, save_path)  # Save the model state
            wait = 0  # Reset early stopping counter
        else:
            wait += 1  # Increment wait counter

        # Check for early stopping condition
        if wait >= maxWait and epoch > minEpoch:
            break

        print(f'bestEpoch:{bestEpoch}')

    # Print the total training time
    endTime = time.time()
    print(f"Training time: {int(endTime - startTime)}s")

    # Save training and validation losses
    # io.savemat('dataset/40ns_test/trainLossPlot_71points_1.mat', {'trainLossPlot': trainLossPlot})
    # io.savemat('dataset/40ns_test/valLossPlot_71points_1.mat', {'valLossPlot': valLossPlot})
    print(f'bestEpoch:{bestEpoch}')
    model.load_state_dict(bestNet)
    model.eval()  # Set the model to evaluation mode

    testLoss = 0.0  # Initialize test loss
    test_predicted_list = []  # Store predictions
    with torch.no_grad():  # Disable gradient calculation
        for data in test_loader:
            feature, label = data  # Get test data
            feature, label = feature.to(device), label.to(device)
            predicted = model(feature.unsqueeze(1))  # Model prediction
            predicted = torch.squeeze(predicted)[:, 20:520]
            label = label.squeeze()
            loss = nn.MSELoss()(predicted, label)  # Calculate test loss
            testLoss += loss.item()
            test_predicted = predicted.cpu().detach().numpy()  # Convert predictions to numpy array
            test_predicted_list.extend(np.ravel(test_predicted))  # Flatten and store predictions

    testLoss = testLoss / len(test_loader)  # Calculate average test loss
    print('test_loss = ', testLoss)

    # Save test results and labels
    test_predicted_list = np.array(test_predicted_list)
    test_label_list = np.load(test_lab_dir, allow_pickle=True)[:, 20:520]
    # np.save('dataset/40ns_test/predictions_and_labels/test_predicted_list_71points_1.npy', test_predicted_list)
    # io.savemat('dataset/40ns_test/predictions_and_labels/test_predicted_list_71points_1.mat', {'predicted_list': test_predicted_list})
    # io.savemat('dataset/40ns_test/predictions_and_labels/label_71points_1.mat', {'test_label_list': test_label_list})

    print('Testing Epoch Finished')

