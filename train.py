#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from tqdm import *
from utils import *
import json
from model import VQAModel



def train(options):
	sess_config = tf.ConfigProto()
	sess_config.gpu_options.allow_growth=False
	sess = tf.InteractiveSession(config=sess_config)

	batch_size = options['batch_size']
	max_epochs = options['max_epochs']
	lr_init = options['learning_rate']
	status_file = options['status_file']


	split = 'train'
	options['input_file_pattern'] = 'data/tf_record/%s/*'%split
	dataset = json.load(open('data/encode_QA/%s.json'%split))
	n_iters_per_epoch = len(dataset['image_id']) // batch_size


	###############
	# build model #
	###############
	logging.info(Fore.GREEN + 'build model ...')
	model = VQAModel(options)
	outputs = model.build()
	t_loss = outputs['loss']
	t_accuracy = outputs['accuracy']
	t_summary = tf.summary.merge_all()

	with tf.variable_scope('Train'):
		t_global_step = tf.Variable(0, name='global_step', trainable=False)
		t_lr = tf.train.exponential_decay(
			learning_rate=lr_init, 
			global_step=t_global_step,
			decay_steps=5e4,
			decay_rate=0.5
			)
		opt_op = tf.contrib.layers.optimize_loss(
			loss=t_loss,
			global_step=t_global_step,
			learning_rate=t_lr,
			optimizer=options['solver']
		)
		
		ema = tf.train.ExponentialMovingAverage(decay=0.9999, num_updates=t_global_step)
		ema_op = ema.apply(tf.trainable_variables())
		with tf.control_dependencies([opt_op]):
			train_op = tf.group(ema_op)
		
		saver = tf.train.Saver(max_to_keep=1, name='Saver')
		save_path = os.path.join(options['ckpt_prefix'], 'best_model')
		train_summary_writer = tf.summary.FileWriter(options['ckpt_prefix'], sess.graph)


	tf.global_variables_initializer().run()

	# Start the queue runners
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
	if options['init_from']:
		logging.info(Fore.GREEN + 'Init model from %s ...'%options['init_from'])
		init_variables = tf.trainable_variables()
		init_variables = [x for x in init_variables if 'logit' not in x.name]
		print ','.join([x.name for x in init_variables])
		init_saver = tf.train.Saver(init_variables, name='Init_Saver')
		init_saver.restore(sess, options['init_from'])

	json_worker_status = OrderedDict()
	json_worker_status['options'] = options
	json.dump(json_worker_status, open(options['status_file'], 'w'))
		
	st_epoch = t_global_step.eval() / n_iters_per_epoch
	for epoch in range(st_epoch, max_epochs):
		print()
		logging.info(Fore.GREEN + 'epoch: %d/%d, lr: %.1E (%.1E)' % (epoch, max_epochs, t_lr.eval(), lr_init))
		for iter in trange(n_iters_per_epoch):
			_, summary, loss, acc = sess.run([train_op, t_summary, t_loss, t_accuracy])
			if iter % options['n_iters_display'] == 0:
				train_summary_writer.add_summary(summary, global_step=t_global_step.eval())

		saver.save(sess, save_path)
	
	train_summary_writer.close()
	coord.request_stop()
	coord.join(threads)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	options = default_options()
	for key, value in options.items():
		parser.add_argument('--%s'%key, dest=key, type=type(value), default=None)
	args = parser.parse_args()
	args = vars(args)
	for key, value in args.items():
		if value != None:
			options[key] = value
			if key == 'ckpt_prefix':
				if not options['ckpt_prefix'].endswith('/'):
					options['ckpt_prefix'] = options['ckpt_prefix'] + '/'
				options['status_file'] = options['ckpt_prefix'] + 'status.json'
	
	set_logging(options)

	work_dir = options['ckpt_prefix']
	if os.path.exists(work_dir) :
		logging.warning(Fore.YELLOW + 'work_dir %s exists! Pls check it.'%work_dir)
	else:
		os.makedirs(work_dir)

	os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu'])
	train(options)

	
