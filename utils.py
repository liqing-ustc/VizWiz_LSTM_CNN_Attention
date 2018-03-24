from collections import OrderedDict
import tensorflow as tf
import pickle as pkl
import numpy as np
import logging
import sys
from time import time

from colorama import init
from colorama import Fore, Back, Style

init(autoreset = True)

def default_options():
	options = OrderedDict()
	options['gpu'] = ''
	options['init_from'] = ''

	# model
	options['num_region'] = 196 # 14x14
	options['img_feat_dim'] = 2048
	options['vocab_size'] = 3804
	options['word_emb_size'] = 512
	options['rnn_size'] = 512
	options['rnn_type'] = 'lstm'
	options['attention_size'] = 512
	options['n_answer_class'] = 3000
	options['reverse_question'] = True
	options['drop_prob'] = 0.5
	options['two_final_fc'] = 1
	options['final_fc_size'] = 1024


	# optimization
	options['solver'] = 'Adam' # 'Adam','RSMProp','SGD', 'Momentum'
	options['batch_size'] = 256
	options['learning_rate'] = 1e-3
	options['reg'] = 1e-6 #regularization strength
	options['init_scale'] = 0.08 # the init scale for uniform, here for initializing word embedding matrix
	options['max_epochs'] = 50
	options['n_iters_display'] = 10
	options['n_eval_per_epoch'] = 1 # number of evaluations per epoch
	options['eval_init'] = False # evaluate the initialized model
	options['shuffle'] = True

	# logging
	options['ckpt_prefix'] = 'checkpoints/'
	options['status_file'] = options['ckpt_prefix'] + 'status.json'
	options['n_iters_display'] = 10
	options['log_level'] = 'debug'

	return options

def set_logging(options):
	level = getattr(logging, options['log_level'].upper(), logging.INFO)
	#fmt = '[%(asctime)s - %(levelname)s - %(module)s: %(lineno)s (%(funcName)s)] %(message)s'
	fmt = '[%(asctime)s-%(module)s: %(lineno)s] %(message)s'
	handler = logging.StreamHandler(sys.stdout)
	logging.basicConfig(format=fmt, level=level, handlers=[handler])








