import tensorflow as tf
from evaluation.metrics import ConfusionMatrix
import logging
import gin
from visualization.grad_cam import grad_cam, superimpose_gradcam
from visualization.transformer_grad_cam import transformer_grad_cam, transformer_superimpose_gradcam
from input_pipeline.preprocessing import preprocess
import matplotlib.pyplot as plt
import seaborn as sns


@gin.configurable
def evaluate(model, ds_test, ds_info, learning_rate, run_paths, last_layer_name, inp_img_path, model_name):
    logging.info("Entering Into Evalution mode")
    conf_mat = ConfusionMatrix()

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), net=model)
    #manager = tf.train.CheckpointManager(ckpt, directory=run_paths["path_ckpts_train"], max_to_keep=10, checkpoint_name= "ckpt_@step")
    ckpt.restore(run_paths).expect_partial()
    if run_paths:
        logging.info("Restored from {}".format(run_paths))

        for idx, (test_images, test_labels) in enumerate(ds_test):
            #template = 'Test Labels: {}'
            #logging.info(template.format(test_labels))
            test_predictions = model(test_images, training=False)
            #template = 'Test Predictions: {}'
            #logging.info(template.format(test_predictions))
            conf_mat.update_state(test_labels, test_predictions)
        
        
        #GRAD CAM PART
        ##To read and expanad dims the image
        image_string = tf.io.read_file(inp_img_path)
        grad_cam_test_image = tf.io.decode_image(image_string, channels=3, expand_animations = False)
        grad_cam_test_image = tf.expand_dims(grad_cam_test_image, axis=0)
        ## To preprocess before sending it to the model
        pre_grad_cam_test_image, _ = preprocess(image = grad_cam_test_image, label = None)
        if model_name!= "transformer_small_idrid":
            ## Heatmap generation function call
            heatmap = grad_cam(pre_grad_cam_test_image, model, last_layer_name)

            #Superimposed heatmap and image function call
            superimposed_img = superimpose_gradcam(heatmap)
            # Save the superimposed image
            cam_path = "/home/RUS_CIP/st180270/dl-lab-22w-team13/diabetic_retinopathy/visualization/" + model_name +"/grad_cam_4.jpg" #provide the path to store the final picture
            superimposed_img.save(cam_path)
        
        else:
            ## Heatmap generation function call
            heatmap = transformer_grad_cam(pre_grad_cam_test_image, model, last_layer_name)

            #Superimposed heatmap and image function call
            superimposed_img = transformer_superimpose_gradcam(heatmap)
            # Save the superimposed image
            cam_path = "/home/RUS_CIP/st180270/dl-lab-22w-team13/diabetic_retinopathy/visualization/" + model_name +"/grad_cam_4.jpg" #provide the path to store the final picture
            superimposed_img.save(cam_path)

        

        #Confusion matrix and other metrics
        cm = conf_mat.result()
        plt.figure(figsize=(2, 2))
        sns.heatmap(cm, annot=True)
        plt.savefig("/home/RUS_CIP/st180270/dl-lab-22w-team13/diabetic_retinopathy/visualization/" + model_name + "/confusion_matrix_4.png")
        template = 'confusion matrix: {}'
        logging.info(template.format(cm))
        prob = conf_mat.other_metrics()
        template = 'Balanced_Accuracy: {:.2f}%, f1_score: {:.2f}%, Unbalanced_Accuracy: {:.2f}%, Specificity: {:.2f}%, Sensitivity: {:.2f}%'
        logging.info(template.format(prob["Balanced_Accuracy"]*100, prob["f1_score"]*100, prob["Unbalanced_Accuracy"]*100, prob["Specificity"]*100, prob["Sensitivity"]*100))
    
    else:
        logging.info("No model found on that checkpoint")
    return None