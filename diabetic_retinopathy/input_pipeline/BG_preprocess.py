import cv2
import numpy as np
import os
import glob
import pandas as pd

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
    cv2.circle(dummy_img, (int(img.shape[1]/2), int(img.shape[0]/2)), int(scale_size*0.9),(1,1,1),-1, 8, 0) 
    
    #Removing 10% of the image to remove the boundary.
    final_img = (img*dummy_img)+(128*(1-dummy_img))
    return final_img

#Crop off the dark portion of the image
def dark_spot_remove(img, beginning_end, end_start):
    final_img = img[:, beginning_end:end_start, :]
    return final_img

def gr_preprocess(image_path, new_dir,  scale_size = 300):
    print(os.path.join(image_path , '/images/test/*jpg'))
    for path in glob.glob(image_path + '/images/test/*jpg'):
        #to extract the final path to store the preprocessed images
        
        print(path)
        #Dataset preprocessing as per the ben graham 
        #print(image)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = radius_scaling(img, scale_size)
        img = local_avg_color_substract(img, scale_size)
        img = boundary_remove(img, scale_size)
        preprocessed_img = dark_spot_remove(img, 30, 707) #crops first 0 to 30 columns and 707 to 737 columns from a given images
        #img.shape [486, 657, 3]
        final_path = os.path.join(new_dir, path.split('/')[-1]) 
        print(final_path)
        cv2.imwrite(final_path, preprocessed_img)
    
if __name__ == "__main__":
    gr_preprocess(image_path = '/home/data/IDRID_dataset', new_dir= '/home/RUS_CIP/st176425/dl-lab-22w-team13/diabetic_retinopathy/IDRID_dataset/images/test/',scale_size = 300)
    print('success')