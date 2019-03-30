import numpy as np
from PIL import Image
from crop_image import crop_image


def process_image(image, mean= np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
        
    #open image in read mode
    image=Image.open(image, mode='r')
    
    #scale image, use smaller dimension as resize argument
    width, height = image.size
    if width>height:
        image.thumbnail((1e6, 256))
    else:
        image.thumbnail((256, 1e6))
        
    #crop image
    
    image=crop_image(image)
    
    #normalize
    image=np.array(image)/255
    image=(image-std)/mean
    
    #reorder dimensions 
    image=image.transpose((2,0,1))
    
    return image