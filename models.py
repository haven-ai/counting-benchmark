
import torch.nn as nn
import torchvision
import torch
from skimage import morphology as morph
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os, tqdm
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import skimage
from lcfcn import lcfcn_loss

from haven import haven_img as hi
from scipy import ndimage
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import cv2
from haven import haven_img
from haven import haven_utils as hu
import torch.utils.model_zoo as model_zoo

# Model
# ===============
class CountingModel(torch.nn.Module):
    def __init__(self, exp_dict, train_set):
        super().__init__()
        self.exp_dict = exp_dict
        self.n_classes = train_set.n_classes
        self.exp_dict = exp_dict

        self.model_base = FCN8_VGG16( n_classes=self.n_classes)

        if self.exp_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam(
                self.model_base.parameters(), lr=self.exp_dict["lr"], betas=(0.99, 0.999), weight_decay=0.0005)

        elif self.exp_dict["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(
                self.model_base.parameters(), lr=self.exp_dict["lr"])

        else:
            raise ValueError

    def train_on_loader(model, train_loader):
        model.train()

        n_batches = len(train_loader)
        train_meter = Meter()
        
        pbar = tqdm.tqdm(total=n_batches)
        for batch in train_loader:
            score_dict = model.train_on_batch(batch)
            train_meter.add(score_dict['train_loss'], batch['images'].shape[0])

            pbar.set_description("Training. Loss: %.4f" % train_meter.get_avg_score())
            pbar.update(1)

        pbar.close()

        return {'train_loss':train_meter.get_avg_score()}

    @torch.no_grad()
    def val_on_loader(self, val_loader, savedir_images=None, n_images=2):
        self.eval()

        n_batches = len(val_loader)
        val_meter = metrics.Meter()
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(tqdm.tqdm(val_loader)):
            score_dict = self.val_on_batch(batch)
            val_meter.add(score_dict['miscounts'], batch['images'].shape[0])
            
            pbar.update(1)

            if savedir_images and i < n_images:
                os.makedirs(savedir_images, exist_ok=True)
                self.vis_on_batch(batch, savedir_image=os.path.join(
                    savedir_images, "%d.jpg" % i))
                
                pbar.set_description("Validating. MAE: %.4f" % val_meter.get_avg_score())

        pbar.close()
        val_mae = val_meter.get_avg_score()
        val_dict = {'val_mae':val_mae, 'val_score':-val_mae}
        return val_dict

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        self.train()

        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        logits = self.model_base.forward(images)
        loss = lcfcn_loss.compute_loss(points=points, probs=logits.sigmoid())
        
        loss.backward()

        self.opt.step()

        return {"train_loss":loss.item()}


    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt":self.opt.state_dict()}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])

    def val_on_batch(self, batch):
        self.eval()
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        logits = self.model_base.forward(images)
        probs = logits.sigmoid().cpu().numpy()

        blobs = lcfcn_loss.get_blobs(probs=probs)

        return {'miscounts': abs(float((np.unique(blobs)!=0).sum() - 
                                (points!=0).sum()))}
        
    @torch.no_grad()
    def vis_on_batch(self, batch, savedir_image):
        self.eval()
        images = batch["images"].cuda()
        points = batch["points"].long().cuda()
        logits = self.model_base.forward(images)
        probs = logits.sigmoid().cpu().numpy()

        blobs = lcfcn_loss.get_blobs(probs=probs)

        pred_counts = (np.unique(blobs)!=0).sum()
        pred_blobs = blobs
        pred_probs = probs.squeeze()

        # loc 
        pred_count = pred_counts.ravel()[0]
        pred_blobs = pred_blobs.squeeze()
        
        img_org = hu.get_image(batch["images"],denorm="rgb")

        # true points
        y_list, x_list = np.where(batch["points"][0].long().numpy().squeeze())
        img_peaks = haven_img.points_on_image(y_list, x_list, img_org)
        text = "%s ground truth" % (batch["points"].sum().item())
        haven_img.text_on_image(text=text, image=img_peaks)

        # pred points 
        pred_points = lcfcn_loss.blobs2points(pred_blobs).squeeze()
        y_list, x_list = np.where(pred_points.squeeze())
        img_pred = hi.mask_on_image(img_org, pred_blobs)
        # img_pred = haven_img.points_on_image(y_list, x_list, img_org)
        text = "%s predicted" % (len(y_list))
        haven_img.text_on_image(text=text, image=img_pred)

        # heatmap 
        heatmap = hi.gray2cmap(pred_probs)
        heatmap = hu.f2l(heatmap)
        haven_img.text_on_image(text="lcfcn heatmap", image=heatmap)
        
        
        img_mask = np.hstack([img_peaks, img_pred, heatmap])
        
        hu.save_image(savedir_image, img_mask)

# Metrics
# ==============
class Meter:
    def __init__(self):
        self.n_sum = 0.
        self.n_counts = 0.

    def add(self, n_sum, n_counts):
        self.n_sum += n_sum 
        self.n_counts += n_counts

    def get_avg_score(self):
        return self.n_sum / self.n_counts


# Network
# ===============
class FCN8_VGG16(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        # PREDEFINE LAYERS
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)
      
        # VGG16 PART
        self.conv1_1 = conv3x3(3, 64, stride=1, padding=100)
        self.conv1_2 = conv3x3(64, 64)
        
        self.conv2_1 = conv3x3(64, 128)
        self.conv2_2 = conv3x3(128, 128)
        
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(256, 256)
        self.conv3_3 = conv3x3(256, 256)

        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(512, 512)
        self.conv4_3 = conv3x3(512, 512)

        self.conv5_1 = conv3x3(512, 512)
        self.conv5_2 = conv3x3(512, 512)
        self.conv5_3 = conv3x3(512, 512)
        
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0)
        
        # SEMANTIC SEGMENTAION PART
        self.scoring_layer = nn.Conv2d(4096, self.n_classes, kernel_size=1, 
                                      stride=1, padding=0)

        self.upscore2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 
                                          kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(self.n_classes, self.n_classes,
                                         kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 
                                    kernel_size=16, stride=8, bias=False)
        
        # Initilize Weights
        self.scoring_layer.weight.data.zero_()
        self.scoring_layer.bias.data.zero_()
        
        self.score_pool3 = nn.Conv2d(256, self.n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.n_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        self.upscore2.weight.data.copy_(get_upsampling_weight(self.n_classes, self.n_classes, 4))
        self.upscore_pool4.weight.data.copy_(get_upsampling_weight(self.n_classes, self.n_classes, 4))
        self.upscore8.weight.data.copy_(get_upsampling_weight(self.n_classes, self.n_classes, 16))

        # Pretrained layers
        pth_url = 'https://download.pytorch.org/models/vgg16-397923af.pth' # download from model zoo
        state_dict = model_zoo.load_url(pth_url)

        layer_names = [layer_name for layer_name in state_dict]

        
        counter = 0
        for p in self.parameters():
            if counter < 26:  # conv1_1 to pool5
                p.data = state_dict[ layer_names[counter] ]
            elif counter == 26:  # fc6 weight
                p.data = state_dict[ layer_names[counter] ].view(4096, 512, 7, 7)
            elif counter == 27:  # fc6 bias
                p.data = state_dict[ layer_names[counter] ]
            elif counter == 28:  # fc7 weight
                p.data = state_dict[ layer_names[counter] ].view(4096, 4096, 1, 1)
            elif counter == 29:  # fc7 bias
                p.data = state_dict[ layer_names[counter] ]


            counter += 1

    def forward(self, x):
        n,c,h,w = x.size()
        # VGG16 PART
        conv1_1 =  self.relu(  self.conv1_1(x) )
        conv1_2 =  self.relu(  self.conv1_2(conv1_1) )
        pool1 = self.pool(conv1_2)
        
        conv2_1 =  self.relu(   self.conv2_1(pool1) )
        conv2_2 =  self.relu(   self.conv2_2(conv2_1) )
        pool2 = self.pool(conv2_2)
        
        conv3_1 =  self.relu(   self.conv3_1(pool2) )
        conv3_2 =  self.relu(   self.conv3_2(conv3_1) )
        conv3_3 =  self.relu(   self.conv3_3(conv3_2) )
        pool3 = self.pool(conv3_3)
        
        conv4_1 =  self.relu(   self.conv4_1(pool3) )
        conv4_2 =  self.relu(   self.conv4_2(conv4_1) )
        conv4_3 =  self.relu(   self.conv4_3(conv4_2) )
        pool4 = self.pool(conv4_3)
        
        conv5_1 =  self.relu(   self.conv5_1(pool4) )
        conv5_2 =  self.relu(   self.conv5_2(conv5_1) )
        conv5_3 =  self.relu(   self.conv5_3(conv5_2) )
        pool5 = self.pool(conv5_3)
        
        fc6 = self.dropout( self.relu(   self.fc6(pool5) ) )
        fc7 = self.dropout( self.relu(   self.fc7(fc6) ) )
        
        # SEMANTIC SEGMENTATION PART
        # first
        scores = self.scoring_layer( fc7 )
        upscore2 = self.upscore2(scores)

        # second
        score_pool4 = self.score_pool4(pool4)
        score_pool4c = score_pool4[:, :, 5:5+upscore2.size(2), 
                                         5:5+upscore2.size(3)]
        upscore_pool4 = self.upscore_pool4(score_pool4c + upscore2)

        # third
        score_pool3 = self.score_pool3(pool3)
        score_pool3c = score_pool3[:, :, 9:9+upscore_pool4.size(2), 
                                         9:9+upscore_pool4.size(3)]

        output = self.upscore8(score_pool3c + upscore_pool4) 

        return output[:, :, 31: (31 + h), 31: (31 + w)].contiguous()

# Utils
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,stride),
                     padding=(padding,padding))

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0)