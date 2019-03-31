# Imports here
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from collections import OrderedDict
from model_select import model_select
import json
import torch.nn as nn
import torch.optim as optim



def load_model(checkpoint_path, lr=0.001):
    
    
    """
    Loads a .pth or .pt file
    assumes user already know what device model is saved to
    
    """ 
    
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model=model_select(checkpoint['arch'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx=checkpoint['class_to_idx']
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    
    return model, optimizer