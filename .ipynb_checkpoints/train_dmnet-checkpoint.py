#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import ast
import logging
import string
import random
import yaml

from datetime import datetime

from mfcnet.mfcnet.model.mfcnet import DMNet
from model.activations import swish
from training.metrics import Metrics
from training.trainer import Trainer
from training.data_container import DataContainer
from training.data_provider import DataProvider


# set up logger
logger = logging.getLogger()

logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='%(asctime)s (%(levelname)s): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('WARN')
tf.autograph.set_verbosity(2)


# In[3]:


with open('../configs/config_dmnet.yaml', 'r') as c:
    config = yaml.safe_load(c)


# In[4]:


for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass


# In[5]:


model_name = config['model_name']

F = config['F']
L = config['L']
K = config['K']
r_cut = config['r_cut']
atoms = config["atoms"]
num_interaction_blocks = config['num_interaction_blocks']

num_train = config['num_train']
num_valid = config['num_valid']
data_seed = config['data_seed']
dataset = config['dataset']
logdir = config['logdir']

num_steps = config['num_steps']
ema_decay = config['ema_decay']

learning_rate = config['learning_rate']
warmup_steps = config['warmup_steps']
decay_rate = config['decay_rate']
decay_steps = config['decay_steps']

batch_size = config['batch_size']
evaluation_interval = config['evaluation_interval']
save_interval = config['save_interval']
restart = config['restart']
comment = config['comment']
target = config['target']


# ***Create directories***

# In[6]:


# Used for creating a random "unique" id for this run
def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

# Create directories
# A unique directory name is created for this run based on the input
if restart is None:
    directory = (logdir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + model_name
                 + "_" + id_generator()
                 + "_" + os.path.basename(dataset)
                 + "_" + '-'.join(target)
                 + "_" + comment)
else:
    directory = restart
logging.info(f"Directory: {directory}")

if not os.path.exists(directory):
    os.makedirs(directory)
best_dir = os.path.join(directory, 'best')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)
log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
best_loss_file = os.path.join(best_dir, 'best_loss.npz')
best_ckpt_file = os.path.join(best_dir, 'ckpt')
step_ckpt_folder = log_dir


# ***Create summary writer and metrics***

# In[7]:


summary_writer = tf.summary.create_file_writer(log_dir)
train = {}
validation = {}
train['metrics'] = Metrics('train', target)
validation['metrics'] = Metrics('val', target)


# *Load Dataset*

# In[8]:


data_container = DataContainer(L, dataset, target, r_cut)

data_provider = DataProvider(L, data_container, num_train, num_valid, batch_size, seed=data_seed, randomized=True)

train['dataset'] = data_provider.get_dataset('train').prefetch(tf.data.experimental.AUTOTUNE)
train['dataset_iter'] = iter(train['dataset'])
validation['dataset'] = data_provider.get_dataset('val').prefetch(tf.data.experimental.AUTOTUNE)
validation['dataset_iter'] = iter(validation['dataset'])


# *Initialize model*

# In[9]:


model = DMNet(F, L, K, r_cut, num_interaction_blocks, atoms)


# *Save/load best recorded loss*

# In[10]:


if os.path.isfile(best_loss_file):
    loss_file = np.load(best_loss_file)
    metrics_best = {k: v.item() for k, v in loss_file.items()}
else:
    metrics_best = validation['metrics'].result()
    for key in metrics_best.keys():
        metrics_best[key] = np.inf
    metrics_best['step'] = 0
    np.savez(best_loss_file, **metrics_best)


# *Initialize trainer*

# In[11]:


trainer = Trainer(model, learning_rate, warmup_steps, decay_steps, decay_rate, ema_decay, max_grad_norm=1000)


# *Set up checkpointing and load latest checkpoint*

# In[12]:


# Set up checkpointing
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer.optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, step_ckpt_folder, max_to_keep=3)

# Restore latest checkpoint
ckpt_restored = tf.train.latest_checkpoint(log_dir)
if ckpt_restored is not None:
    ckpt.restore(ckpt_restored)


# *Training loop*

# In[13]:


with summary_writer.as_default():
    steps_per_epoch = int(np.ceil(num_train / batch_size))

    if ckpt_restored is not None:
        step_init = ckpt.step.numpy()
    else:
        step_init = 1
    for step in range(step_init, num_steps + 1):
        # Update step number
        ckpt.step.assign(step)
        tf.summary.experimental.set_step(step)

        # Perform training step
        trainer.train_on_batch(train['dataset_iter'], train['metrics'])

        # Save progress
        if (step % save_interval == 0):
            manager.save()

        # Evaluate model and log results
        if (step % evaluation_interval == 0):

            # Save backup variables and load averaged variables
            trainer.save_variable_backups()
            trainer.load_averaged_variables()

            # Compute results on the validation set
            for i in range(int(np.ceil(num_valid / batch_size))):
                trainer.test_on_batch(validation['dataset_iter'], validation['metrics'])

            # Update and save best result
            if validation['metrics'].mean_mae < metrics_best['mean_mae_val']:
                metrics_best['step'] = step
                metrics_best.update(validation['metrics'].result())

                np.savez(best_loss_file, **metrics_best)
                model.save_weights(best_ckpt_file)

            for key, val in metrics_best.items():
                if key != 'step':
                    tf.summary.scalar(key + '_best', val)
                
            epoch = step // steps_per_epoch
            logging.info(
                f"{step}/{num_steps} (epoch {epoch + 1}):"
                f"Loss: train={train['metrics'].loss:.6f}, val={validation['metrics'].loss:.6f};"
                f"logMAE: train={train['metrics'].mean_log_mae:.6f}, "
                f"val={validation['metrics'].mean_log_mae:.6f}"
            )

            train['metrics'].write()
            validation['metrics'].write()

            train['metrics'].reset_states()
            validation['metrics'].reset_states()

            # Restore backup variables
            trainer.restore_variable_backups()





