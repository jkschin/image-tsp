from tspconv import MyNet, CustomDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import copy
import cv2
import os
import numpy as np
import torch.nn.functional as F
import src.visualization.visualization as viz

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = 0
CITY = 1
PATH = 2

img_transform = transforms.Compose([
    transforms.ToTensor(),
])
batch_size = 10


expt1 = "expt/1-1150-20220110-140212"
expt20 = "expt/20-1150-20220109-144909"
expt30 = "expt/30-1150-20220109-144910"
expt100= "expt/100-1150-20220109-144921"
expt200= "expt/200-1150-20220109-144932"

folders = [expt1, expt20, expt30, expt100, expt200]

for expt_folder in folders:
    weights_path = os.path.join(expt_folder, "weights.pth")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ss = models.densenet161(pretrained=True)
    model = MyNet(my_pretrained_model=model_ss)
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    for num_cities in [20, 30, 100, 200]:
        k = 10000
        output_folder = os.path.join(expt_folder, "test-%d" %num_cities)
        dataset = CustomDataset(num_cities, "test", transform=img_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for data in dataloader:
            print("%s: Loaded Batch for %d cities" %(expt_folder, num_cities))
            inp, _ = data
            if torch.cuda.is_available():
                inp = inp.cuda()
            # ===================forward=====================
            # print(inp.dtype)
            # print(inp.shape)
            output = model(inp)
            a = np.array(torch.argmax(output.cpu().data, dim=1))
            for l in range(batch_size):
                b = a[l, :, :]
                c = np.stack((b, ) * 3, -1) 
                c[np.where((c == [1,1,1]).all(axis=2))] = viz.RED
                c[np.where((c == [2,2,2]).all(axis=2))] = viz.GREEN
                # for i in range(512):
                #     for j in range(512):
                #         if np.all(c[i, j, :] == [1,1,1]):
                #           c[i,j,:] = RED
                #         elif np.all(c[i, j, :] == [2,2,2]):
                #           c[i,j,:] = GREEN
                # img1 = np.array(orig_inp[l])
                # img2 = np.array(orig_out[l])
                # vis = np.concatenate((img1, img2, c), axis=1)
                cv2.imwrite(os.path.join(output_folder, "output_%05d.png" %k), c)
                k += 1
