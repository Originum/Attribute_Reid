# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import math
from net import *

model_dict = {
    'resnet50_joint'  :  ResNet50_joint,
}
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/root/dataset/Market-1501/threshold/',type=str, help='./test_data')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--name', default='resnet50_joint', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--multi', default=False, action='store_true', help='use multiple query' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--stride', default=2, type=int, help='stride')

opt = parser.parse_args()

model_dir = os.path.join('./checkpoints', opt.dataset, opt.name)
nclasses = 751
attr_num = 30
print("stride",opt.stride)


str_ids = opt.gpu_ids.split(',')
name = opt.name
test_dir = opt.test_dir


gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_dir = test_dir


image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','positive','negative']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=0) for x in ['gallery','positive','negative']}
use_gpu = torch.cuda.is_available()
fea_cat = False

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join(model_dir,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor().cuda()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        if fea_cat == True:
            ff = torch.FloatTensor(n,542).zero_().cuda()
        else:
            ff = torch.FloatTensor(n,512).zero_().cuda()
    
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                pred_attr, outputs = model(input_img)
                if fea_cat == True:
                    outputs = torch.cat((outputs, pred_attr), 1)
                ff += outputs
       
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data), 0)
    return features

def get_id(img_path):
    labels = []
    frames = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        frame = filename[0:16]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        frames.append(frame)
    return labels, frames


gallery_path = image_datasets['gallery'].imgs
positive_path = image_datasets['positive'].imgs
negative_path = image_datasets['negative'].imgs

gallery_label,gallery_frame = get_id(gallery_path)
positive_label,positive_frame = get_id(positive_path)
negative_label,negative_frame = get_id(negative_path)
######################################################################
# Load Collected data Trained model
print('-------test-----------')

model_structure = model_dict[name](attr_num, nclasses, opt.stride)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.classifier_reid.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    positive_feature = extract_feature(model,dataloaders['positive'])
    negative_feature = extract_feature(model,dataloaders['negative'])
    
#######################################################################
# Evaluate
def evaluate(qf,qfr,gf,gl,gfr):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    junk_index = np.argwhere(gl == -1)
    frame_index = np.argwhere(gfr == qfr)
    good_index = np.setdiff1d(index, frame_index, assume_unique=True)
    good_index = np.setdiff1d(good_index, junk_index, assume_unique=True)
    return score[good_index[0]]

######################################################################
#query_feature = query_feature.cuda()
#gallery_feature = gallery_feature.cuda()

#print(query_label)
positive_dist = []
negative_dist = []
for i in range(len(positive_label)):
    dist = evaluate(positive_feature[i],positive_frame[i],gallery_feature,gallery_label,gallery_frame)
    positive_dist.append(dist)

for i in range(len(negative_label)):
    dist = evaluate(negative_feature[i],negative_frame[i],gallery_feature,gallery_label,gallery_frame)
    negative_dist.append(dist)

print('{} positive:'.format(len(positive_dist)))
print(positive_dist)
print('{} negative:'.format(len(negative_dist)))
print(negative_dist)
# Save to Matlab for check
result = {'positive':positive_dist,'negative':negative_dist}
scipy.io.savemat('threshold.mat',result)

