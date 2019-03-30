# Imports here
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets
import torchvision.models as models 
from torch.optim import lr_scheduler
from model_select import model_select
import datetime


def save_model(model, optimizer, device='cpu'):
    """
    Saves a model and optimizer state dict to device
    model saved to today's date by default
    
    """
    
    
    #default save to CPU
    model.to(device)
    
    path = datetime.datetime.today().strftime('%Y_%m_%d')
    path=path+'.pth'
    
    #check for .pth or .pt extension
    torch.save({
        'model_state_dict': model.state_dict(),
        'arch': model.arch,
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print("model saved to... ", path)
