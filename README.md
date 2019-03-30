# RosieMcRoseFace

Multiclass classification on PyTroch with command line application using python's argparse module.

The dataset contains images of flowers belonging to 102 different categories. 
The images were acquired by searching the web and taking pictures. There are a
minimum of 40 images for each category.

To download the flower images:
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html


Required packages:

- numpy
- torch
- import torchvision
- matplotlib
-  time
-  os
-  copy
-  PIL
-  collections
-  seaborn 


## Training

The training function can be called by running the train.py script as shown below


'''console
$ python3 train.py -h
usage: train.py [-h] [--model MODEL] [--lr LR] [--epochs N] [--no-cuda]
                [--save-model] [--path SAVE]

PyTorch flower classification TRAINING script

optional arguments:
  -h, --help     show this help message and exit
  --model MODEL  select a model: vgg16, vgg19, alexnet (default = vgg16)
  --lr LR        learning rate (default = 0.001)
  --epochs N     number of training epochs (default = 10)
  --no-cuda      disables training on CUDA if called
  --save-model   Saves current model to cpu as YYYY_MM_DD.pth if called
  --path SAVE    select filename, pth extenstion automatically added (default
                 = saved_model)

'''

## Prediction

The prediction function can be called by running the predict.py script as shown below

'''console  
$ python3 predict.py -h
usage: predict.py [-h] [--model MODEL] [--no-cuda] [--topk tk] [--img IMG]
                  [--label_map lablels] [--lr LR]

PyTorch flower classification PREDICT script

optional arguments:
  -h, --help           show this help message and exit
  --model MODEL        select model path (default = None)
  --no-cuda            disables training on CUDA if called
  --topk tk            Top k predictions (default = 5)
  --img IMG            image to predict (default
                       =flowers/train/100/image_07930.jpg )
  --label_map lablels  picks a json file to append names to flowers (default =
                       None)
  --lr LR              learning rate (default = 0.001)
  
  '''
