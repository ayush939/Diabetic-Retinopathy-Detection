import gin
import wandb
import tensorflow as tf
import logging
from utils.utils_misc import Callback_EarlyStopping

'''
### HIGHLIGHTS OF THIS FILE
1. In order to obtain the prediction result, F1_score, TP, FP, TN, FN on certain training and validation steps,  the train step and val step sequeues are defining under the for-loop instead of calling the train_step and val_step tf functions. 
'''
@gin.configurable
class Trainer(object):
<<<<<<< HEAD
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval, learning_rate, after_step_save):
        
        # Loss objective
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        #self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
=======
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval, learning_rate):
        
        
        # Loss objective
        # self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #self.loss_object = tf.compat.v1.losses.sparse_softmax_cross_entropy()
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

        #self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        #self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        # Optimizer and Learning Rate Schedule
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=1000, decay_rate=0.96) # learning__rate * (decay_rate^(step/decay_steps)) Note: step increase everytime. step is increased everytime apply_gradient is called
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        #To store the loss values for Custom Early Stopping Callback
        self.LossList = []
        self.early_stopping_flag = False
        
        #https://keras.io/api/metrics/classification_metrics/

        self.train_true_positive = tf.keras.metrics.TruePositives(thresholds=0.5)
        self.train_true_negative = tf.keras.metrics.TrueNegatives(thresholds=0.5)
        self.train_false_positive =  tf.keras.metrics.FalseNegatives(thresholds=0.5)
        self.train_false_negative =  tf.keras.metrics.FalsePositives(thresholds=0.5)


        self.val_true_positive = tf.keras.metrics.TruePositives(thresholds=0.5)
        self.val_true_negative = tf.keras.metrics.TrueNegatives(thresholds=0.5)
        self.val_false_positive =  tf.keras.metrics.FalseNegatives(thresholds=0.5)
        self.val_false_negative =  tf.keras.metrics.FalsePositives(thresholds=0.5)

        self.train_precision = tf.keras.metrics.Precision(name='train_precision', thresholds=0.5)
        self.val_precision = tf.keras.metrics.Precision(name='val_precision', thresholds=0.5)

        self.train_recall = tf.keras.metrics.Recall(name='train_recall', thresholds=0.5)
        self.val_recall = tf.keras.metrics.Recall(name='val_recall', thresholds=0.5)
        
        '''
        if name == 'eyepacs':
            #self.train_weights = [2.56, 0.62]
            #self.val_weights = [2.56, 0.62]
            self.train_weights = None
            self.val_weights = None

        elif name == 'idrid':

            self.train_weights = ds_info[0] #calculated the weights value in dataset python file
            self.val_weights = ds_info[1]
        '''

        self.weights = tf.constant([0.79, 1.34]) 
        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.after_step_save = after_step_save

        # Summary Writer
        self.writer  = tf.summary.create_file_writer(self.run_paths["path_model_id"])

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
<<<<<<< HEAD
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=self.run_paths["path_ckpts_train"], max_to_keep=50, checkpoint_name= "ckpt_@step")
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            logging.info("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            logging.info("Initializing from scratch.")
    
    '''
=======
        self.manager = tf.train.CheckpointManager(self.ckpt, self.run_paths["path_ckpts_train"], max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
<<<<<<< HEAD
            loss = self.loss_object(labels, predictions)

=======
            class_weights = self.weights
            weights = tf.gather(class_weights, labels)
            loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, predictions, weights=weights)
            
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy.update_state(labels, predictions)
        
        self.train_true_positive.update_state(labels, predictions)
        self.train_true_negative.update_state(labels, predictions)
        self.train_false_positive.update_state(labels, predictions)
        self.train_false_negative.update_state(labels, predictions)

        self.train_precision.update_state(labels, predictions)
        self.train_recall.update_state(labels, predictions)


    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        class_weights = self.weights
        weights = tf.gather(class_weights, labels)
        t_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, predictions, weights)
        
        self.val_loss(t_loss)
        self.val_accuracy.update_state(labels, predictions)

        self.val_true_positive.update_state(labels, predictions)
        self.val_true_negative.update_state(labels, predictions)
        self.val_false_positive.update_state(labels, predictions)
        self.val_false_negative.update_state(labels, predictions)

        self.val_precision.update_state(labels, predictions)
        self.val_recall.update_state(labels, predictions)
    '''
    
    def train(self):
        #wandb.watch(self.model, log="all", log_freq=100)
        for idx, (images, labels) in enumerate(self.ds_train):
<<<<<<< HEAD
            #template = 'train labels: {}'
            #logging.info(template.format(labels))
            step = idx + 1
            ### self.train_step(images, labels)

            # The Training Step starts here. 
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = self.model(images, training=True)
                #template = 'Predictions: {}'
                #logging.info(template.format(predictions))
                loss = self.loss_object(labels, predictions)

            #To stop the program if there is no change for some patience time (Custom Early Stopping Function)
            self.LossList.append(loss.numpy())
            if step%500==0:
                self.early_stopping_flag = Callback_EarlyStopping(self.LossList, min_delta=0.0001, patience=100)
=======
            #print(images.shape)
            #inp = input("Check")
            step = idx + 1
            self.train_step(images, labels)
            wandb.log({"step": step, "train_loss": self.train_loss.result()})
            if step % self.log_interval == 0:
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            #template = 'Steps: {}'
            #logging.info(template.format(step))
            
            self.train_loss(loss)
            self.train_accuracy.update_state(labels, predictions)
            
            self.train_true_positive.update_state(labels, predictions)
            self.train_true_negative.update_state(labels, predictions)
            self.train_false_positive.update_state(labels, predictions)
            self.train_false_negative.update_state(labels, predictions)

            self.train_precision.update_state(labels, predictions)
            self.train_recall.update_state(labels, predictions)

            if step % self.log_interval == 0 and step >= self.after_step_save:
                
                # Calculating the training F1 score for addition metrics
                train_f1_score = 2 * (self.train_recall.result() * self.train_precision.result())/(self.train_recall.result() + self.train_precision.result())
                
                # Updating the training metrics parameters to WandB site for visualization
                wandb.log({"step": step, "train_loss": self.train_loss.result()})
                wandb.log({"step": step, "train_accuracy": self.train_accuracy.result()})
                wandb.log({"step": step, "train_f1_score": train_f1_score})

                # Reset val metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()
  
                self.val_precision.reset_states()
                self.val_recall.reset_states()

                self.val_true_positive.reset_states()
                self.val_false_positive.reset_states()
                self.val_true_negative.reset_states()
                self.val_false_negative.reset_states()

                # The Validation Step Starts here
                val_steps = 0
                for val_images, val_labels in self.ds_val:
<<<<<<< HEAD
                    #template = 'Steps: {}'
                    #logging.info(template.format(val_steps))
                    #template = 'Val labels: {}'
                    #logging.info(template.format(val_labels))
                    ### self.val_step(val_images, val_labels)
                    
                    val_predictions = self.model(val_images, training=False)
                    #template = 'Val Predictions: {}'
                    #logging.info(template.format(val_predictions))
                    t_loss = self.loss_object(val_labels, val_predictions)

                    self.val_loss(t_loss)
                    self.val_accuracy.update_state(val_labels, val_predictions)

                    self.val_true_positive.update_state(val_labels, val_predictions)
                    self.val_true_negative.update_state(val_labels, val_predictions)
                    self.val_false_positive.update_state(val_labels, val_predictions)
                    self.val_false_negative.update_state(val_labels, val_predictions)

                    self.val_precision.update_state(val_labels, val_predictions)
                    self.val_recall.update_state(val_labels, val_predictions)
                    val_steps+=1

                # Calculating the validationg F1 score for addition metrics
                val_f1_score = 2 * (self.val_recall.result() * self.val_precision.result())/(self.val_recall.result() + self.val_precision.result())
                
                # Updating the validation metrics parameters to WandB site for visualization
                wandb.log({"step": step, "val_loss": self.val_loss.result()})
                wandb.log({"step": step, "val_accuracy": self.val_accuracy.result()})
                wandb.log({"step": step, "val_f1_score": val_f1_score})
                
                template = 'Step {}, Loss: {}, Accuracy: {}, F1_score: {}, Validation Loss: {}, Validation Accuracy: {}, Validation F1_score: {}, Train TP: {}, Train FP: {}, Train TN: {}, Train FN: {}, Val TP: {}, Val FP: {}, Val TN: {}, Val FN: {}'
=======
                    self.val_step(val_images, val_labels)
                    wandb.log({"step": step, "val_loss": self.val_loss.result()})
                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             train_f1_score *100,

                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100, 
                                             val_f1_score * 100,

                                             
                                            self.train_true_positive.result(),
                                            self.train_false_positive.result(),
                                            self.train_true_negative.result(),
                                            self.train_false_negative.result(),

                                            self.val_true_positive.result(),
                                            self.val_false_positive.result(),
                                            self.val_true_negative.result(),
                                            self.val_false_negative.result()

                                             ))
               
                # Write summary to tensorboard
                with self.writer.as_default():
                    tf.summary.scalar("Training loss", self.train_loss.result(), step=step)
<<<<<<< HEAD
                    tf.summary.scalar("Validation loss", self.val_loss.result(), step=step)
=======
                    tf.summary.scalar("Validatoin loss", self.val_loss.result(), step=step)

>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                self.train_precision.reset_states()
                self.train_recall.reset_states()

                self.train_true_positive.reset_states()
                self.train_false_positive.reset_states()
                self.train_true_negative.reset_states()
                self.train_false_negative.reset_states()

                yield self.val_accuracy.result().numpy()
        

            if step % self.ckpt_interval == 0 and step >= self.after_step_save:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
<<<<<<< HEAD
                self.manager.save(checkpoint_number = step)
=======
                self.manager.save()
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc

            if (step % self.total_steps == 0) or self.early_stopping_flag:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
<<<<<<< HEAD
                self.manager.save(checkpoint_number = step)
=======
                self.manager.save()
>>>>>>> 3cf6961b390ce177f0ec2cd49d1016e3fd9b56bc
                return self.val_accuracy.result().numpy()
