import logging
import gin

import ray
from ray import tune

from input_pipeline.datasets import load
from models.architectures import *
from train import Trainer
from utils import utils_params, utils_misc


def train_func(config):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings))

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['/home/RUS_CIP/st176425/dl-lab-22w-team13/diabetic_retinopathy/configs/config.gin'], bindings) # change path to absolute path of config file
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    model = new_model()

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


ray.init(num_cpus=10, num_gpus=1)
analysis = tune.run(
    train_func, num_samples=2, resources_per_trial={"cpu": 10, "gpu": 1},
    config={
        "prepare.batch_size": tune.choice([64, 128]),
        "Trainer.learning_rate": tune.loguniform(0.001, 0.1),
        "new_model.dropout_rate": tune.uniform(0, 0.9)
    })

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
