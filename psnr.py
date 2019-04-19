import numpy as np
import math

def psnr(hr_image, sr_image):
    #assume RGB image
    hr_image_data = np.array(hr_image)
    sr_image_data = np.array(sr_image).astype('float32')
    
    diff = sr_image_data - hr_image_data
    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(255.0/rmse)
