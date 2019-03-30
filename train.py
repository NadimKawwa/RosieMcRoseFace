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






### DEFINE DIRECTORY ###

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#store the directory in a dict
directory={'train': train_dir,
          'valid': valid_dir}


#### Define  transforms for the training and validation sets ####
trans_mean=[0.485, 0.456, 0.406]
trans_std= [0.229, 0.224, 0.225]

#how many samples per batch to load
batch_size=32

train_transforms=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, hue=.05, saturation=.05),
    transforms.RandomRotation((-90,90)),
    transforms.RandomVerticalFlip(),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=(-45,45),shear=(-45,45)),
    transforms.Pad(5),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=trans_mean, std=trans_std)
])

#no augmentation on valid data

valid_transforms=transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=trans_mean, std=trans_std)])

data_transforms={'train': train_transforms,
                'valid': valid_transforms}



#### Load the datasets with ImageFolder ####
image_datasets={key: datasets.ImageFolder(directory[key],
                                          transform=data_transforms[key]) for key in directory}




#### Define the dataloaders ####
batch_size=32

dataloaders= {key: torch.utils.data.DataLoader(image_datasets[key], batch_size=batch_size,
                                              shuffle=True) for key in directory}




dataset_sizes = {key: len(image_datasets[key]) for key in directory}
class_names = image_datasets['train'].classes


#### TRAINING FUNCTION ####


def train_model(model, optimizer, num_epochs=20, device='cpu'):
    since = time.time()

    # default criterion
    criterion= nn.NLLLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model=model.to(device)

    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                #optimizer.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]: #NB: calling a global variable inside a function
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            #Save the loss for inference
            if phase=='train':
                train_losses.append(epoch_loss)
            else:
                test_losses.append(epoch_loss)

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, test_losses


#### MAIN FUNCTION PARSER ####

def main():

    parser=argparse.ArgumentParser(description="PyTorch flower classification TRAINING script")
    #select model from list
    parser.add_argument('--model', type=str, default='vgg16', metavar='MODEL',
                       help='select a model: vgg16, vgg19, alexnet (default = vgg16)')

    #define learning rate
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                       help='learning rate (default = 0.001)')

    #define number of epochs
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                       help='number of training epochs (default = 10)')

    #cuda or no cuda
    parser.add_argument('--no-cuda',action='store_true', default=False,
                       help='disables training on CUDA if called')

    #save or not save
    parser.add_argument('--save-model', action='store_true', default=False,
                       help="Saves current model to cpu as YYYY_MM_DD.pth if called")

    #saved path
    parser.add_argument('--path', type=str, default='saved', metavar='SAVE',
                       help='select filename, pth extenstion automatically added (default = saved_model)')


    args = parser.parse_args()

    #see if cuda can be chosen and user selection
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #set device
    if use_cuda:
        device=torch.device('cuda')
        print("training on cuda")
    else:
        device = torch.device("cpu")
        print("training on cpu")


    #load model selection
    model=model_select(args.model)


    #set up optimizer with learning rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    num_epochs=args.epochs

    model_ft, train_losses, test_losses =train_model(model, optimizer, num_epochs, device)

    
    #list for future mapping
    model_ft.class_to_idx = image_datasets['train'].class_to_idx
    

    if args.save_model:
        save_model(model_ft, optimizer, device)


#### LAUNCH main() ####
if __name__ == '__main__':
    main()


