__author__ = 'schin'

import cv2
import torch
from torch import nn
from torchvision import transforms, models
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from collections import Counter
import numpy as np
from torchinfo import summary


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = 0
CITY = 1
PATH = 2


def to_img(x):
    x = torch.argmax(x, dim=1)
    print(torch.unique(x))
    print(Counter(torch.flatten(x).tolist()))
    x *= 126
    # x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.type(torch.FloatTensor)
    # print(x)
    # print(torch.max(x))
    # print(x.shape)
    # print(x.dtype)
    x = x.view(x.size(0), 1, image_size[0], image_size[1])
    # print(x.dtype)
    # print(torch.max(x))
    return x

class CustomDataset(Dataset):
    def __init__(self, num_cities, mode, combined=False, transform=None):
        self.num_cities = num_cities
        self.transform = transform
        self.mode = mode
        self.combined = combined

    def __len__(self):
        if torch.cuda.is_available() and self.mode == "train":
            if self.combined:
                return 40000
            else:
                return 10000
        elif torch.cuda.is_available() and self.mode == "test":
            return 1000
        else:
            return 100

    def transform_img(self, img, colors, classes):
        new_img = np.zeros(img.shape[0:2])
        zipped = zip(colors, classes)
        for color, classs in zipped:
            indexes = np.where((img == color).all(axis=2))
            new_img[indexes] = classs
        return new_img

    def transform_inp(self, img):
        colors = [BLACK, RED, WHITE]
        classes = [BACKGROUND, CITY, PATH]
        img = self.transform_img(img, colors, classes)
        return img

    def transform_out(self, img):
        colors = [BLACK, RED, GREEN]
        classes = [BACKGROUND, CITY, PATH]
        img = self.transform_img(img, colors, classes)
        return img

    def __getitem__(self, idx):
        num_cities = self.num_cities
        if self.mode == "test":
            idx += 10000
        if self.combined:
            q, r = idx // 10000, idx % 10000
            cities = [20, 30, 100, 200]
            num_cities = cities[q]
            idx = r
        inp_name = os.path.join("data/images/%d/%s/input" %(num_cities, self.mode), "input_%05d.png" %idx)
        inp = cv2.imread(inp_name)
        inp = self.transform_inp(inp)
        inp = torch.as_tensor(inp, dtype=torch.int64)
        inp = F.one_hot(inp)
        inp = torch.transpose(inp, 1, 2)
        inp = torch.transpose(inp, 0, 1)
        inp = torch.squeeze(inp)
        inp = inp.type(torch.FloatTensor)

        out_name = os.path.join("data/images/%d/%s/output" %(num_cities, self.mode), "output_%05d.png" %idx)
        out = cv2.imread(out_name)
        out = self.transform_out(out)
        out = torch.as_tensor(out, dtype=torch.int64)
        out = out.type(torch.LongTensor)
        # test = torch.as_tensor(out, dtype=torch.int64)
        # test = F.one_hot(test)
        # test = torch.transpose(test, 1, 2)
        # test = torch.transpose(test, 0, 1)
        # test = torch.squeeze(test)
        # test = test.type(torch.FloatTensor)
        return inp, out


class MyNet(nn.Module):
    def __init__(self, my_pretrained_model):
        image_size = (512, 512) # hardcoded this for a quick fix
        super(MyNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.u1 = nn.Sequential(
            nn.BatchNorm2d(2208),
            nn.ReLU(True),
            nn.Conv2d(2208, 32, 1),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.u2 = nn.Sequential(
            nn.BatchNorm2d(2112),
            nn.ReLU(True),
            nn.Conv2d(2112, 32, 1),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.u3 = nn.Sequential(
            nn.BatchNorm2d(768),
            nn.ReLU(True),
            nn.Conv2d(768, 32, 1),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.u4 = nn.Sequential(
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(384, 32, 1),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.last = nn.Sequential(
            nn.Conv2d(130, 3, 1, stride=1),
        )

    def forward(self, x):
        # for name, layer in self.pretrained.named_modules():
        #     print(name, layer)
        #     x = layer(x)
        outputs = []
        x_orig = x
        for ii, model in enumerate(list(self.pretrained.features)):
            if ii in [11]:
                continue
            x = model(x)
            if ii in [2, 4, 6, 8, 10]:
                outputs.append(x)
        u1 = self.u1(outputs[-1])
        u2 = self.u2(outputs[-2])
        u3 = self.u3(outputs[-3])
        u4 = self.u4(outputs[-4])
        # 1: Copies the cities and path to the last layer
        x = torch.cat([u1, u2, u3, u4, x_orig[:, 1:, :, :]], 1)
        x = self.last(x)
        # a = self.u1(x)
        # b = self.u2(x)
        # x = self.upsample(self.pretrained.features.relu0)
        # x = self.last(x)
        return x

from time import localtime, strftime
import sys
if __name__ == "__main__":
    # label = binascii.b2a_hex(os.urandom(15))[0:10].decode("utf-8")
    label = strftime("%Y%m%d-%H%M%S", localtime())
    # Params to Tune
    # 1. Weights 2. Num Epochs 3. Num Cities 4. Combined
    weights = torch.FloatTensor([1, 1, 50])
    num_epochs = 200
    num_cities = int(sys.argv[1])
    combined = True

    expt = "%s-%s-%s" %(num_cities, "1150", label)
    num_classes = 3
    output_dir = os.path.join("expt", expt)
    # This cannot be RGB format.
    image_size = (512, 512)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sets Batch Size
    if torch.cuda.is_available():
        batch_size = 16
    else:
        batch_size = 1
    learning_rate = 1e-3
    print("Batch Size:", batch_size)

    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(num_cities, "train", combined=combined, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model_ss = models.densenet161(pretrained=True)
    # for name, param in model_ss.named_parameters():
    # if not "classifier" in name:
    # param.requires_grad = False
    mynet = MyNet(my_pretrained_model=model_ss)
    model = mynet.to(device)
    summary(model, input_size=(batch_size, 3, image_size[0], image_size[1]))
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform
    model.apply(init_weights)
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        print("Epoch: %d" %epoch)
        total_loss = 0
        for data in dataloader:
            print("Loaded Batch on", expt)
            inp, out = data
            if torch.cuda.is_available():
                inp, out = inp.cuda(), out.cuda()
            # ===================forward=====================
            # print(inp.dtype)
            # print(inp.shape)
            output = model(inp)
            loss = criterion(output, out)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        total_loss += loss.data
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, total_loss))
        # if epoch % 10 == 0:
        print("Image written")
        pic = to_img(output.cpu().data)
        save_image(pic, os.path.join(output_dir, "image_%d.png" %epoch))
        torch.save(model.state_dict(), os.path.join(output_dir, 'weights.pth'))
