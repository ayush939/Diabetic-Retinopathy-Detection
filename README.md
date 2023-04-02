
# How to run the code
Steps to follow to run the projects:

## Diabetic Retinopathy: 
- The main.py file has flag variables to select the models to train, the graham preprocessing image generation, and the models to evaluate. Please select the models as per the model hint provided inn the file. 
- Upon the flag variable selection, Each model has been designed to select its required config file to operate automatically. 
- The checkpoints, confusion matrix and grad cam of the models are designed to store on a respective folders as per the flag variable selection. 
NOTE: sometime error may occur due to the fault in folder creation, then manually creation of folders required. 
- The main.py file has a BT_graham_not_available flag variable when True can create the graham preprocessing image and store them on the same folder as the original images. Use this variable only once in the project.

## HAPT
- Set the 'mode' and 'model_name' flag variables to enable training/evaluation and lstm/gru in 'main.py' python file and run it.
- By default it will create the tfrecord files from the path of the raw dataset mentioned in config files. 
- Copy the trainied model in 'checkpoints' folder and run evaluation. 

Caution: The path shown in the config files are dedicated to the creator's path. Do change them. 


