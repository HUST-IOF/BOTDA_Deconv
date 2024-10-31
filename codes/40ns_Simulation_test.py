import time
import torch
from torch.nn import init
from torchvision import transforms

import os
import numpy as np
import torch.nn as nn
from torch.utils import data
from torch import nn,optim
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler
from torch.optim import *
from scipy import io
import torchvision.models.resnet
# from model2 import resnet18
from model3 import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestDataset(data.Dataset):
    def __init__(self, lab_dir, fea_dir, transform=None):
        self.transform = transform
        self.lab_dir = lab_dir
        self.fea_dir = fea_dir
        self.fea_list = np.load(fea_dir, allow_pickle=True)
        self.lab_list = np.load(lab_dir, allow_pickle=True)

    def __getitem__(self, idx):
        feature = self.fea_list[:, :, idx]  # 71*n
        feature = feature.squeeze()
        feature = torch.FloatTensor(feature)
        label = self.lab_list[idx, 20:520]  # n*500
        label = torch.FloatTensor(label)
        return feature, label

        # feature = self.feature[idx, :]
        # label = self.label[idx, :]

    def __len__(self):
        dataset_len = int(np.size(self.lab_list, 0))
        return dataset_len


# main
if __name__ == "__main__":
    startTime = time.time()
    model = resnet34().to(device)
    checkpoint = torch.load('./models/resnet34_5point_40ns_71.pth')
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)

    test_lab_dir = f'./datasets\\40ns_test\\simulation_data_71points_BFS.npy'
    test_fea_dir = f'./datasets\\40ns_test\\simulation_data_71points_BGS.npy'
    test_dataste = TestDataset(test_lab_dir,test_fea_dir)

    # loader
    test_loader = torch.utils.data.DataLoader(test_dataste,batch_size = 16,shuffle=False)

    # run

    print('Testing...')
    model.eval()
    testLoss = 0.0
    testLossPlot = []
    test_predicted_list = []
    test_label_list = []
    with torch.no_grad():
        for data in test_loader:
            feature, label = data
            feature, label = feature.to(device), label.to(device)
            predicted = model(feature.unsqueeze(1))
            predicted = predicted.squeeze(1)
            # print(predicted.size())
            predicted = predicted.squeeze(1)[:, 20:520]
            # print(predicted.size())
            label = label.squeeze(1)
            loss = nn.MSELoss()(predicted, label)
            testLoss += loss.item()
            test_predicted = predicted.cpu().detach().numpy()
            test_label = label.cpu().detach().numpy()
            test_predicted = list(np.ravel(test_predicted))
            test_label = list(np.ravel(test_label))
            test_predicted_list.extend(test_predicted)
            test_label_list.extend(test_label)
        testLoss = testLoss / len(test_loader)
        test_predicted_list = np.array(test_predicted_list)
        test_label_list = np.array(test_label_list)
    endTime = time.time()
    print(f"训练耗时:{int(endTime - startTime)}s")

    # Define the x-axis
    x = np.arange(0.1, 50.1, 0.1)  # Create an array of values from 0.1 to 50.1 with a step of 0.1 for the x-axis

    # Create the first plot
    plt.figure(figsize=(2.3, 1.6), dpi=300)  # Set the figure size and resolution to 2.3x1.6 inches and 300 dpi
    plt.plot(x, test_label_list[53300:53800],
             linewidth=1)  # Plot the data from test_label_list (slice from index 53300 to 53800) with a line width of 1
    plt.plot(x, test_predicted_list[53300:53800], linewidth=1,
             linestyle='--')  # Plot the predicted data (same slice) with a dashed line style and line width of 1
    # Uncomment the following line to set the x-axis limits to a specific range
    # plt.xlim([30, 31.2])
    plt.xlabel('Fiber length (m)', fontsize=6,
               fontweight='bold')  # Set the x-axis label with specified font size and weight
    plt.ylabel('BFS (GHz)', fontsize=6, fontweight='bold')  # Set the y-axis label with specified font size and weight
    legend = plt.legend(['Label', 'Output'], fontsize=8,
                        loc='northeast')  # Add a legend with specified labels, font size, and location
    # Attempt to adjust the linewidth and markersize in the legend (note that markersize adjustment may have limited effect)
    for legline, origline in zip(legend.get_lines(), plt.gca().get_lines()):
        legline.set_linewidth(origline.get_linewidth())  # Set the legend line width to match the original line width
        legline.set_markersize(
            6)  # Attempt to set the marker size in the legend (this may not have a visible effect for lines without markers)
    plt.gca().set_fontname(
        'Cambria')  # Attempt to set the global font name to Cambria (this may require additional configuration depending on your environment)
    plt.tight_layout(pad=0.01)  # Adjust the padding between and around subplots to make the layout more compact
    plt.close()  # This line is redundant because plt.show() will be called immediately after; it closes the figure before displaying it, which is not usually desired
    plt.show()  # Display the plot on the screen (without saving it to a file)
