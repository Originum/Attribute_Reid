# !/usr/local/bin/python3
import os
import time
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datafolder.folder import Train_Dataset
from net import *

######################################################################
# Settings
# --------
use_gpu = True
Sigma = 1
Lambda = 0.5
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
model_dict = {
    'resnet18'  :  ResNet18_nFC,
    'resnet34'  :  ResNet34_nFC,
    'resnet50'  :  ResNet50_nFC,
    'densenet'  :  DenseNet121_nFC,
    'resnet50_softmax'  :  ResNet50_nFC_softmax,
    'resnet50_single'  :  ResNet50_single,
    'resnet50_joint'  :  ResNet50_joint,
}

######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data-path', default='/root/dataset/', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--model', default='resnet50_joint', type=str, help='model')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
parser.add_argument('--which-epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--num-workers', default=1, type=int, help='num_workers')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--continuing', action='store_true', help='continue the training' )

args = parser.parse_args()

assert args.dataset in dataset_dict.keys()
assert args.model in model_dict.keys()

data_dir = args.data_path
model_dir = os.path.join('./checkpoints', args.dataset, args.model)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
print("Sigma:",Sigma)
print("Lambda:",Lambda)
print("batch_size:",args.batch_size)
print("stride:",args.stride)
print("erasing_p:",args.erasing_p)
print("warm_epoch:",args.warm_epoch)
print("lr:",args.lr)
print("num_epoch:",args.num_epoch)
######################################################################
# Function
# --------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(model_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if use_gpu:
        network.cuda()


def load_network(network):
    save_path = os.path.join(model_dir,'net_%s.pth'%args.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network
######################################################################
# Draw Curve
#-----------
x_epoch = []
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join(model_dir, 'train.jpg'))


######################################################################
# DataLoader
# ---------
image_datasets = {}
image_datasets['train'] = Train_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                        train_val='train', erasing_p = args.erasing_p, SIZE = (384, 128))
image_datasets['val'] = Train_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                      train_val='query', SIZE = (384, 128))
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

images, indices, labels, ids, cams, names = next(iter(dataloaders['train']))

num_label = image_datasets['train'].num_label()
num_id = image_datasets['train'].num_id()
labels_list = image_datasets['train'].labels()
distribution = image_datasets['train'].distributions()

distribution = torch.from_numpy(distribution).float()
weights = torch.exp(-distribution/(Sigma*Sigma))

######################################################################
# Model and Optimizer
# ------------------
model = model_dict[args.model](num_label, num_id, args.stride)
if args.continuing:
	model = load_network(model)
	print("continue the training")
else:
	print("the new training")
if use_gpu:
    model = model.cuda()
    weights = weights.cuda()
# loss
criterion_attr = nn.BCELoss(weight = weights)
criterion_reid = nn.CrossEntropyLoss()
# optimizer


ignored_params = (list(map(id, model.classifier_attribute.parameters()))+list(map(id, model.classifier_reid.parameters())))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer = torch.optim.SGD([
         {'params': base_params, 'lr': 0.1*args.lr},
         {'params': model.classifier_attribute.parameters(), 'lr': args.lr},
         {'params': model.classifier_reid.parameters(), 'lr': args.lr}
     ], weight_decay = 5e-4, momentum = 0.9, nesterov = True)

#optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 5e-4, nesterov = True)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 40, gamma = 0.1)


######################################################################
# Training the model
# ------------------
def train_model(model, criterion_attr, criterion_reid, optimizer, scheduler, num_epochs, Lambda):
    since = time.time()

    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/args.batch_size)*args.warm_epoch # first 5 epoch

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_loss_id = 0.0
            running_corrects_id = 0

            # Iterate over data.
            for count, data in enumerate(dataloaders[phase]):
                # get the inputs
                images, indices, labels, ids, cams, names = data
                # wrap them in Variable
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                    indices = indices.cuda()
                images = images
                labels = labels.float()
                indices = indices

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs_attr, outputs_reid = model(images)

                attr_loss = criterion_attr(outputs_attr, labels)
                reid_loss = criterion_reid(outputs_reid, indices)

                joint_loss = Lambda * reid_loss + (1 - Lambda) * attr_loss


                # backward + optimize only if in training phase
                if epoch<args.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    joint_loss *= warm_up


                if phase == 'train':
                    joint_loss.backward()
                    optimizer.step()

                preds = torch.gt(outputs_attr, torch.ones_like(outputs_attr)/2 ).data
                # statistics
                running_loss += attr_loss.item()
                running_corrects += torch.sum(preds.byte() == labels.data.byte()).item() / num_label
                #print('step : ({}/{})  |  loss : {:.4f}'.format(count*args.batch_size, dataset_sizes[phase], label_loss.item()))
                running_loss_id += reid_loss.item()
                v, i = torch.max(outputs_reid, 1)
                running_corrects_id += torch.sum(indices == i).item()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_loss_id = running_loss_id / len(dataloaders[phase])
            epoch_acc_id = running_corrects_id / dataset_sizes[phase]

            if phase == 'train':
                print('{} Loss: {:.4f} Acc: {:.4f} ID_Loss: {:.4f} ID_Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_loss_id, epoch_acc_id))
            else:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)
            else:
                scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')


######################################################################
# Main
# -----
model = train_model(model, criterion_attr, criterion_reid, optimizer, exp_lr_scheduler,
                    num_epochs = args.num_epoch, Lambda = Lambda)