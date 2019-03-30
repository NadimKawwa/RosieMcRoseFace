import numpy as np
from PIL import Image




def crop_image(pil_image, new_width=224, new_height=224):

    width, height = pil_image.size 
 
    left = (width - new_width)/2
    bottom = (height - new_height)/2
    top = (height + new_height)/2
    right = (width + new_width)/2
    
    box=(left, bottom, right, top)
    

    cropped_image=pil_image.crop(box)
    
    return cropped_image