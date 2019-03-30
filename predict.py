import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from collections import OrderedDict
from workspace_utils import active_session
import seaborn as sns
import argparse
from model_select import model_select
from save_model import save_model
from load_model import load_model
from process_image import process_image
from crop_image import crop_image
import json


#### Predict function ####

def predict(image_path, model,topk=5, device='cpu', label_map=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
        
    #process image for PyTorch mode
    image=process_image(image_path)
    
    #convert torch to tensor
    tensor_image= torch.from_numpy(image).type(torch.FloatTensor)
    
    #return a tensor with all the dimensions of input size of 1 removed 
    squeeze_image=tensor_image.unsqueeze(0)
    
    #move image to device
    squeeze_image=squeeze_image.to(device)
    
    #move model to device
    model = model.to(device)
    
    #set to eval mode
    model.eval()
    
    #extract probability
    with torch.no_grad():
        proba=torch.exp(model.forward(squeeze_image))
        #return k largest elements of input tensor with dimensions
        top_probas, top_labels= proba.topk(topk)
        
    #move the probas and labels to cpu
    top_probas= top_probas.to('cpu')
    top_labels= top_labels.to('cpu')
    
    #convert probas and labels to numpy array
    top_probas= top_probas.detach().numpy().tolist()[0]
    top_labels= top_labels.detach().numpy().tolist()[0]
    
    #convert numeric label to categirocal label
    idx_to_class={val: key for key, val in model.class_to_idx.items()}
    
    top_labels=[idx_to_class[int(lab)] for lab in top_labels]
    
    if label_map:
        with open(label_map, 'r') as f:
            cat_to_name = json.load(f)
        top_flowers= [cat_to_name[idx_to_class[int(label)]] for label in top_labels]
        for i, (proba, flower) in enumerate(zip(top_probas, top_flowers)):
            print("{}. {}: {:.3f}".format(i, flower, proba))
        
    else:
        for i, (proba, label) in enumerate(zip(top_probas, top_labels)):
            print("{}. {}: {:.3f}".format(i, label, proba))

    
    

#### MAIN FUNCTION PARSER ####

def main():

    parser=argparse.ArgumentParser(description="PyTorch flower classification PREDICT script")
    #select model from list
    parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                       help='select model path (default = None)')

    #cuda or no cuda
    parser.add_argument('--no-cuda',action='store_true', default=False,
                       help='disables training on CUDA if called')

    #save topk
    parser.add_argument('--topk', type=float, default=5, metavar='tk',
                       help='Top k predictions (default = 5)')
    
    #select model from list
    parser.add_argument('--img', type=str, default='flowers/train/100/image_07930.jpg', metavar='IMG',
                       help='image to predict (default =''flowers/train/100/image_07930.jpg'' )')
    
    #json for names
    parser.add_argument('--label_map',type=str, default=None, metavar='lablels',
                       help='picks a json file to append names to flowers (default = None)')
    
    #define learning rate
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                       help='learning rate (default = 0.001)')




    args = parser.parse_args()

    #see if cuda can be chosen and user selection
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #set device
    if use_cuda:
        device=torch.device('cuda')
        print("Predicting on cuda")
    else:
        device = torch.device("cpu")
        print("Predicting on cpu")
        
        
    model, optimizer =load_model(args.model, args.lr)
    
    
    predict(args.img, model,args.topk, device, args.label_map)
    
            
    
#### LAUNCH main() ####
if __name__ == '__main__':
    main()
    
    


