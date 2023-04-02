import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, num_classes=2, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.conf_mat = self.add_weight(name="conf_mat", shape=(num_classes, num_classes),
                                  dtype=tf.float32, initializer="zeros")
        self.true_positive = tf.keras.metrics.TruePositives(thresholds=0.5)
        self.true_negative = tf.keras.metrics.TrueNegatives(thresholds=0.5)
        self.false_positive =  tf.keras.metrics.FalseNegatives(thresholds=0.5)
        self.false_negative =  tf.keras.metrics.FalsePositives(thresholds=0.5)
        self.precision = tf.keras.metrics.Precision(name='precision', thresholds=0.5)
        self.recall = tf.keras.metrics.Recall(name='recall', thresholds=0.5)

    def update_state(self, y_true, y_pred):
        # y_true = tf.squeeze(y_true, axis=-1)
        #y_pred = tf.argmax(y_pred, axis=1)
        #matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2)
        self.true_positive.update_state(y_true, y_pred)
        self.true_negative.update_state(y_true, y_pred)
        self.false_positive.update_state(y_true, y_pred)
        self.false_negative.update_state(y_true, y_pred)
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

        #self.conf_mat.assign_add(matrix)

    def result(self):
        matrix = tf.constant([[self.true_positive.result().numpy(), self.false_positive.result().numpy()], [self.false_negative.result().numpy(), self.true_negative.result().numpy()]], shape=(2,2), dtype=tf.float32)
        self.conf_mat.assign_add(matrix)
        return self.conf_mat

    def other_metrics(self):
        tp = self.true_positive.result()
        tn = self.true_negative.result()
        fp = self.false_positive.result()
        fn = self.false_negative.result()
        precision = self.precision.result()
        recall = self.recall.result()

        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn) 

        balanced_acc = (specificity + sensitivity)/2
        f1_score = 2 * (recall * precision)/(recall + precision)
        unbalanced_acc = (tp+tn) / (tp+tn+fn+fp)

        return {"Balanced_Accuracy":  balanced_acc.numpy(), "f1_score":  f1_score.numpy(), "Unbalanced_Accuracy":  unbalanced_acc.numpy(), "Specificity": specificity.numpy(), "Sensitivity":sensitivity.numpy()}

if __name__ == "__main__":
    a = np.random.randint(0, 2, (1000, 1))
    b = tf.random.uniform((1000, 1))
    metrics_cm = ConfusionMatrix()
    metrics_cm.update_state(a, b)
    cm = metrics_cm.result()
    template = 'Balanced_Accuracy: {:.2f}%, f1_score: {:.2f}%, Unbalanced_Accuracy: {:.2f}%'
    prob = metrics_cm.other_metrics()
    logging.info(template.format(prob["Balanced_Accuracy"]*100, prob["f1_score"]*100, prob["Unbalanced_Accuracy"]*100))