import gin
import wandb
import logging
from absl import app, flags
<<<<<<< HEAD
import tensorflow as tf
=======
import input_pipeline.BG_preprocess as pp
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets, ben_graham_preprocessing
from utils import utils_params, utils_misc
<<<<<<< HEAD
from models.architectures import vgg_like
#from models.architectures import transformer_backbone, ViT_classifier, head, 
from models.architectures import ViT_Classifier_Viz_Only
from models.architectures import resnet_like
from models.architectures import standard_models
from models.architectures import ensemble_model
import wandb

'''
### HIGHLIGHTS OF THIS FILE
1. Different FLAGS.model conditions are created to eary-run the required model. Also the checkpoints and logs are stored as per the defined model path
2. Different gin-configs files are created for each type of models which are accessed as per the requirements.
3. FLAGS.train is used under the FLAGS.model conditions, to avoid folder creation and log creation while evaluation. 
'''
=======
from models.architectures import *
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc

FLAGS = flags.FLAGS

flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')

# At the first time, make it to 'False' and then make it True
flags.DEFINE_boolean('BT_graham_not_available', default=False, help = 'Specify whether to preprocessed Benjamin Graham images are available or not.')

# For the different models
flags.DEFINE_string('model', default = 'transformer_model', help = 'choose from [custom_model, standard_pretrained_model, transformer_model, ensemble_pretrained_model]')
flags.DEFINE_string('custom_model', default = 'resnet_like', help = 'choose from [vgg_like, resnet_like]')
flags.DEFINE_string('standard_pretrained_model', default = 'resnet50', help = 'choose from [resnet50, inceptionV3, xception]')
flags.DEFINE_string('transformer_model', default = 'transformer_small_idrid', help = 'choose from [transformer_small_idrid, transformer_medium_eyepacs]')


def main(argv):
    # generate preprocessing as per the ben graham methods and save them in the folder
    if FLAGS.BT_graham_not_available:
        train_image_path = "/home/RUS_CIP/st180270/datasets/IDRID_dataset/images/train/*.jpg"
        #test_image_path = "/home/RUS_CIP/st180270/datasets/IDRID_dataset/images/test/*.jpg"
        ben_graham_preprocessing.gr_preprocess(train_image_path, scale_size=300, data="idrid")
        logging.info(f"Done Graham Preprocessing...")
    
    #initiating wandb
    wandb.login(key='481adc65f4b182df9a11e068e08a3de5da829dba')

    ### DEFINING CUSTOM MODELS - MAIN File
    if FLAGS.model == "custom_model":
        
        #For Grad Cam during evaluate
        last_layer_name_dict = {"vgg_like": "conv2d_9", "resnet_like" : "add_3"}
        last_layer_name = last_layer_name_dict[FLAGS.custom_model]
        evaluation_model_name = FLAGS.custom_model

        if FLAGS.train:

            path = "/home/RUS_CIP/st180270/dl-lab-22w-team13/diabetic_retinopathy/latest_experiments/custom_model/"+ FLAGS.custom_model

            # generate folder structures
            run_paths = utils_params.gen_run_folder(path)

            # set loggers
            utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)


            utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # gin-config
        gin.parse_config_files_and_bindings(['configs/custom_model_config.gin'], [])

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = datasets.load()

        if FLAGS.custom_model == "vgg_like":
            model = vgg_like()
            #logging.info(model.summary())
            if FLAGS.train:
                # Initializing the WandB site Project
                wandb.init(project="VGGlike", config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        elif FLAGS.custom_model == "resnet_like":
            model = resnet_like()
            #logging.info(model.summary())
            if FLAGS.train:
                # Initializing the WandB site Project
                wandb.init(project="RESNETlike", config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
    
    ### DEFINING STANDARD PRETRAINED MODELS - MAIN File
    elif FLAGS.model == "standard_pretrained_model":
        
        #For Grad Cam during evaluate
        last_layer_name_dict = {"resnet50" : "conv5_block3_out", "inceptionV3":"mixed10", "xception":"block14_sepconv2_act"}
        last_layer_name = last_layer_name_dict[FLAGS.standard_pretrained_model]
        evaluation_model_name = FLAGS.standard_pretrained_model

        #gin-config
        gin.parse_config_files_and_bindings(['configs/standard_pretrained_model_config.gin'], [])

        if FLAGS.train:
            path = "/home/RUS_CIP/st180270/dl-lab-22w-team13/diabetic_retinopathy/latest_experiments/standard_pretrained_model/"+ FLAGS.standard_pretrained_model

            # generate folder structures
            run_paths = utils_params.gen_run_folder(path)

            # set loggers
            utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

            utils_params.save_config(run_paths['path_gin'], gin.config_str())

<<<<<<< HEAD
            # Initializing the WandB site Project
            wandb.init(project="Standard_Pretrained_Model", config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = datasets.load()

        model = standard_models(model_name=FLAGS.standard_pretrained_model)

    ### DEFINING TRANSFORMER MODEL - MAIN File
    elif FLAGS.model == "transformer_model":
        # gin-config
        #gin.parse_config_files_and_bindings(['configs/transformer_model_config.gin'], [])
        #gin.parse_config_files_and_bindings(['configs/medium_eyepacs_transformer_model_config.gin'],[])
        gin.parse_config_files_and_bindings(['configs/only_viz_transformer_model_config.gin'], [])

        #For Grad Cam during evaluate
        last_layer_name_dict = {"transformer_small_idrid": "layer_normalization", "transformer_medium_eyepacs" : "layer_normalization"}
        last_layer_name = last_layer_name_dict[FLAGS.transformer_model]
        evaluation_model_name = FLAGS.transformer_model

        if FLAGS.train:
            path = "/home/RUS_CIP/st180270/dl-lab-22w-team13/diabetic_retinopathy/latest_experiments/transformer_model/"+ FLAGS.transformer_model

            # generate folder structures
            run_paths = utils_params.gen_run_folder(path)

            # set loggers
            utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)


            utils_params.save_config(run_paths['path_gin'], gin.config_str())

            # Initializing the WandB site Project
            wandb.init(project=FLAGS.transformer_model, config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = datasets.load()

        #model = ViT_classifier(transformer_backbone=transformer_backbone, head=head)
        model = ViT_Classifier_Viz_Only()


    elif FLAGS.model == "ensemble_pretrained_model":
        
        #gin-config
        gin.parse_config_files_and_bindings(['configs/ensemble_model_config.gin'], [])

        #For Grad Cam during evaluate
        last_layer_name = "average"
        evaluation_model_name = FLAGS.model

        if FLAGS.train:
            path = "/home/RUS_CIP/st180270/dl-lab-22w-team13/diabetic_retinopathy/latest_experiments/ensemble_model"

            # generate folder structures
            run_paths = utils_params.gen_run_folder(path)

            # set loggers
            utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

            utils_params.save_config(run_paths['path_gin'], gin.config_str())

            # Initializing the WandB site Project
            wandb.init(project="Ensemble_Model", config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = datasets.load()

        model = ensemble_model(standard_models = standard_models)
         
=======
    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())
    # pp.gr_preprocess(image_path = '/home/data/IDRID_dataset', new_dir= '/home/RUS_CIP/st176425/dl-lab-22w-team13/diabetic_retinopathy/IDRID_dataset/images/train/')

    #initiating wandb
    wandb.login(key='f08340aaaae01c503d46f68656705261fa9d4ae9')
    
    config = { 
        "learning_rate": 0.01,
        "momentum":0.0,
        "resblocks" : [1, 1, 1, 1],
        "batch_size" : 128,
        "Optimizer" : "Adam",
        "Dense units": [64, 32],
        "dropout": 0.7
        }

    wandb.init(project="DL-Lab", config=config)
    
    
    
    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()
    
    logging.info("Data Loaded")
    
    # model
    # model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
    #backbone_ = backbone()
    #head = HEAD()
    #model = full_model(backbone=backbone, HEAD=Head)
    
    model = new_model()
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
    if FLAGS.train:
        # To restore custom model pretrained and save using checkpoints
        #run_paths["path_ckpts_train"] = till .data
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths) 
        for _ in trainer.train():
            continue

    else:
        evaluate(model = model,
                 ds_test = ds_test,
                 ds_info = ds_info,
                 last_layer_name = last_layer_name, 
                 model_name = evaluation_model_name)
    

if __name__ == "__main__":
    app.run(main) 