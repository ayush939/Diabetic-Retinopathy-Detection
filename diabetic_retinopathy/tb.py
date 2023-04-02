import tensorflow as tf
from tensorboard import main as tb
tf.compat.flag.FLAG.logdir = "/experiments/run_2022-11-29T15-07-31-226120/"
tb.main()