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
from model2 import resnet34
from scipy.io import loadmat

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# class TestDataset(data.Dataset):
#     def __init__(self,lab_dir,fea_dir, transform=None):
#         self.transform = transform
#         self.lab_dir = lab_dir
#         self.fea_dir = fea_dir
#         self.fea_list = np.load(fea_dir, allow_pickle=True)
#         self.lab_list = np.load(lab_dir, allow_pickle=True)
#
#     def __getitem__(self,idx):
#         feature = self.fea_list[:,:,idx]#101*n
#         feature = feature.squeeze()
#         feature = torch.FloatTensor(feature)
#
#         label = self.lab_list[idx,20:500]#n*520
#         label = torch.FloatTensor(label)
#         return feature,label
#
#         # feature = self.feature[idx, :]
#         # label = self.label[idx, :]
#
#     def __len__(self):
#         dataset_len = int(np.size(self.lab_list,0))
#         return dataset_len

class TestDataset1(data.Dataset):
    def __init__(self,fea_dir, transform=None):
        self.transform = transform
        self.fea_dir = fea_dir
        self.fea_list = np.load(fea_dir, allow_pickle=True)
        self.n = 400
    def __getitem__(self,idx):
        n = self.n
        feature = self.fea_list[:,idx*500+n:(idx*500+n)+540]#
        feature = feature.squeeze()
        feature = feature / np.max(feature)
        feature = torch.FloatTensor(feature)
        return feature

    def __len__(self):
        dataset_len = (np.size(self.fea_list, 1)-20-self.n) // 500
        # dataset_len = (np.size(self.fea_list, 1) - 20) // 501
        return dataset_len


# main
if __name__ == "__main__":
    startTime = time.time()
    model = resnet34().to(device)
    checkpoint = torch.load('./models/resnet34_5point_40ns_71.pth')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    n = 0
    test_fea_dir = f'./datasets\\Experiment_data\\BGS40ns'+str(n+1)+'.npy'
    test_dataste = TestDataset1(test_fea_dir)

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
            feature = data
            feature = feature.to(device)
            predicted = model(feature.unsqueeze(1))
            predicted = predicted.squeeze(1)
            # print(predicted.size())
            predicted = predicted.squeeze(1)[:, 20: 520]
            test_predicted = predicted.cpu().detach().numpy()
            test_predicted = list(np.ravel(test_predicted))
            test_predicted_list.extend(test_predicted)
        test_predicted_list = np.array(test_predicted_list)
        print('test_predicted_list.shape：', test_predicted_list.shape)
        print('test_predicted_list:', test_predicted_list)

    endTime = time.time()
    print(f"Time:{int(endTime - startTime)}s")


#load data
BFS5 = np.load('./results/figure/Experimental_results/BFS5ns.npy')
BFS5 = BFS5[333:]
BFS_LCF = np.load('./results/figure/Experimental_results/BFS40ns6.npy')
BFS_LCF = BFS_LCF / 1000
BFS_LCF = BFS_LCF[339:]
BFS40 = test_predicted_list* (10.89 - 10.81) + 10.81

# plot
fig, ax = plt.subplots()
fig.set_size_inches(8.6, 6)
fig.canvas.manager.set_window_title('Figure 1')

ax.plot(np.arange(len(BFS_LCF)) * 0.1, BFS_LCF, linewidth=1)
ax.plot(np.arange(len(BFS5)) * 0.1, BFS5 / 10 ** 3, linewidth=1)
ax.plot(np.arange(BFS40.shape[0]) * 0.1, BFS40, linestyle='-.', linewidth=1, color=[0.4660, 0.6740, 0.1880])

legend_handles, legend_labels = ax.get_legend_handles_labels()
legend = ax.legend(legend_handles, ['40 ns pulse', '45/40 ns DPP', 'CNN'],
                   prop={'family': 'Cambria', 'size': 8}, loc='northeast', frameon=False)
for text in legend.get_texts():
    text.set_horizontalalignment('left')

ax.set_xlim(4815, 4850)
ax.set_ylim(10.83, 10.875)
ax.set_xlabel('Fiber length (m)', fontsize=8, fontweight='bold', family='Cambria')
ax.set_ylabel('BFS (GHz)', fontsize=8, fontweight='bold', family='Cambria')
ax.tick_params(axis='both', which='major', labelsize=8, fontweight='bold', family='Cambria')

fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0, hspace=0)
plt.show()






