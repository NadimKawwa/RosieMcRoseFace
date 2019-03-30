# Imports here
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from collections import OrderedDict


def model_select(model, n_classes=102, pretrained=True):
    
    """
    Loads a pretrained model from list:
    - "vgg16"
    - "vgg19"
    - "alexnet"
    
    Select number of classes, default=102
    
    Model pretrained by default
    
    """
    
    if model=='vgg16':
        model = models.vgg16(pretrained=pretrained)
        
        #freeze parameters to stop backprop
        for param in model.parameters():
            param.requires_grad=False
        
        
        #change the classifer to adapt to current needs
        classifier =nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, 1024)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(1024, n_classes)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier=classifier
        model.arch='vgg16'
        print("vgg16 selected")
        
        
    elif model=='vgg19':
        model = models.vgg19(pretrained=pretrained)
        
        #freeze parameters to stop backprop
        for param in model.parameters():
            param.requires_grad=False
        
        classifier =nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, 1024)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(1024, n_classes)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier=classifier
        model.arch='vgg19'
        print("vgg19 selected")

    elif model=='alexnet':
        
        model = models.alexnet(pretrained=pretrained)
        
        #freeze parameters to stop backprop
        for param in model.parameters():
            param.requires_grad=False
        
        classifier =nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(9216, 4096)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, 1024)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(1024, n_classes)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier=classifier
        model.arch='alexnet'
        print("alexnet selected")
    else:
        print("please pick correct model: vgg16, vgg19, alexnet...terminating script")
        pass
        
        
    return model



