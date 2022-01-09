from tspconv import MyNet, CustomDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import copy
import cv2
import os
import numpy as np
import torch.nn.functional as F

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
expt_folder = "expt/100-1120-20220108-214021"
prefix = "-".join(expt_folder.split("/")[1].split("-")[0:2])
num_cities = int(expt_folder.split("/")[1].split("-")[0])
output_folder = os.path.join(expt_folder, "test-%s" %prefix)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
weights_path = os.path.join(expt_folder, "weights.pth")
dataset = CustomDataset(num_cities, "test", transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ss = models.densenet161(pretrained=True)
model = MyNet(my_pretrained_model=model_ss)
model.load_state_dict(torch.load(weights_path))
model = model.to(device)
model.eval()

k = 10000
for data in dataloader:
    print("Loaded Batch")
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
        for i in range(512):
            for j in range(512):
                if np.all(c[i, j, :] == [1,1,1]):
                  c[i,j,:] = RED
                elif np.all(c[i, j, :] == [2,2,2]):
                  c[i,j,:] = GREEN
        # img1 = np.array(orig_inp[l])
        # img2 = np.array(orig_out[l])
        # vis = np.concatenate((img1, img2, c), axis=1)
        cv2.imwrite(os.path.join(output_folder, "output_%05d.png" %k), c)
        k += 1
