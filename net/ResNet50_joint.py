import torch
from torch import nn
from torch.nn import init
from torchvision import models


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)



# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_bottleneck = 512):
        super(ClassBlock, self).__init__()


        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.6)]
        add_block += [nn.Linear(num_bottleneck, output_dim)]
        add_block += [nn.Sigmoid()]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        self.classifier = add_block

    def forward(self, x):
        x = self.classifier(x)
        return x



class ClassBlock_reid(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock_reid, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x


class ResNet50_joint(nn.Module):
    def __init__(self, class_num, id_num):
        super(ResNet50_joint, self).__init__()
        self.model_name = 'resnet50_joint'
        self.class_num = class_num
        self.id_num = id_num

        model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft
        self.num_ftrs = 2048
        num_bottleneck = 512
        self.classifier_attribute = ClassBlock(self.num_ftrs, self.class_num, num_bottleneck)
        self.classifier_reid = ClassBlock_reid(self.num_ftrs + self.class_num, self.id_num, 0.6)
        #for c in range(self.class_num):
        #    self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, num_bottleneck) )

    def forward(self, x):
        x = self.features(x)
        pred_attr = self.classifier_attribute(x)

        joint = torch.cat((x, pred_attr), 1)
        pred_reid = self.classifier_reid(joint)
        #for c in range(self.class_num):
        #    if c == 0:
        #        pred = self.__getattr__('class_%d' % c)(x)
        #    else:
        #        pred = torch.cat((pred, self.__getattr__('class_%d' % c)(x) ), dim=1)
        return pred_attr, pred_reid