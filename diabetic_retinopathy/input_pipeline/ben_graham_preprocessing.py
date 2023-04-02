import cv2
import numpy as np
import glob
import pandas as pd
import logging

#Resize the images to have same radius defined by scale_size
def radius_scaling(img, scale_size):
    #To extract only the width of the image shape (4288,)
    img_dummy = img[int(img.shape[0]/2),:,:].sum(1)

    #To extract the number of the pixel that the eye image has excluuding the black spots
    r = (img_dummy>img_dummy.mean()/10).sum()/2
    #To extract the scaling factor
    s = scale_size*(1/r)
    
    #To resize the image with scale_size as radius
    img = cv2.resize(img, None, fx=s, fy=s)
    
    return img

#Substract the local average color and map it to grey
def local_avg_color_substract(img, scale_size):
    alpha, beta, lambd = 4, -4, 128 #values selected as per the benjamin suggestions

    #Merging the blur image and the original image with suitable alpha and beta and offsetting to lambd
    local_mean_color_img=cv2.addWeighted(img, alpha, cv2.GaussianBlur(img ,(0,0) , scale_size/(scale_size/10)), beta, lambd)

    return local_mean_color_img

#Remove the boundary effect
def boundary_remove(img, scale_size):
    #Create a img with complete black same size as the image
    dummy_img = np.zeros(img.shape) 
    
    #Extract the eye circle size and make a circle in the dummy image
    cv2.circle(dummy_img, (int(img.shape[1]/2), int(img.shape[0]/2)), int(scale_size*0.85),(1,1,1),-1, 8, 0) 
    
    #Removing 10% of the image to remove the boundary.
    final_img = (img*dummy_img)+(128*(1-dummy_img))
    return final_img

#Crop off the dark portion of the image
def dark_spot_remove(img, beginning_end, end_start, data):
    if data=="idrid":
        img = img[:, beginning_end:end_start, :]
    return img

def gr_preprocess(test_image_path, scale_size = 300, data="idrid"):
    #for path in glob.glob(train_image_path)+glob.glob(test_image_path):
    for path in glob.glob(test_image_path):
        #to extract the final path to store the preprocessed images
        final_path = path.split("/")
        final_path[-1] = "preprocessed_" + final_path[-1]
        final_path = "/".join(final_path)
        logging.info(f"Preparing dataset {final_path}...")
        #Dataset preprocessing as per the ben graham 
        #print(image)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = radius_scaling(img, scale_size)
        img = local_avg_color_substract(img, scale_size)
        img = boundary_remove(img, scale_size)
        preprocessed_img = dark_spot_remove(img, beginning_end= 30, end_start=707, data=data) #crops first 0 to 30 columns and 707 to 737 columns from a given images
        #img.shape [486, 657, 3]
        cv2.imwrite(final_path, preprocessed_img)
    return None