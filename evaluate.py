#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from tqdm import *
from utils import *
from model import VQAModel
import cPickle as pkl
import json

def evaluate_options():
	options = {}
	options['ckpt_prefix'] = 'saved_model_trainall/'
	options['split'] = 'val' # 'val', 'test'
	options['out_dir'] = 'results'
	options['gpu'] = ''
	options['save_prob'] = 1
	return options


def evaluate(options):
	split = options['split']

	sess_config = tf.ConfigProto()
	sess_config.gpu_options.allow_growth=False
	sess = tf.InteractiveSession(config=sess_config)

	adict_file = 'data/create_vocab/answer2answer_id.json'
	adict = json.load(open(adict_file))
	reverse_adict = {v: k for k, v in adict.items()}

	options['input_file_pattern'] = 'data/tf_record/%s/*'%options['split']
	logging.info(Fore.GREEN + 'build model ...')
	model = VQAModel(options, mode='eval')
	outputs = model.build()

	dataset = json.load(open('data/encode_QA/%s.json'%options['split']))
	val_count = len(dataset['image_id'])
	batch_size = options['batch_size']
	n_iters = int(np.ceil(float(val_count) / batch_size))

	with tf.variable_scope('Train'):
		ema = tf.train.ExponentialMovingAverage(0.9999)
		ema_op = ema.apply(tf.trainable_variables())
	saver = tf.train.Saver(ema.variables_to_restore())
	model_path = tf.train.latest_checkpoint(options['ckpt_prefix'])
	saver.restore(sess, model_path)

	# Start the queue runners
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	logging.info(Fore.GREEN + 'Evaluating model %s on %s ...'%(model_path, split))
	val_loss_list = []
	val_acc_list = []
	predict_labels = {}
	prob = {}
	for _ in trange(n_iters):
		iter_outputs = sess.run(outputs)
		val_loss_list.append(iter_outputs['loss'] * iter_outputs['preds'].shape[0])
		val_acc_list.append(iter_outputs['accuracy'] * iter_outputs['preds'].shape[0])
		predict_labels.update(dict(zip(iter_outputs['image_id'].tolist(), iter_outputs['preds'])))
		prob.update(dict(zip(iter_outputs['image_id'].tolist(), iter_outputs['prob'])))

	ave_val_loss = sum(val_loss_list) / float(val_count)
	ave_val_acc = sum(val_acc_list) / float(val_count)

	# convert label to answer
	predict_answers = []
	for image_id, predict_label in predict_labels.items():
		answer = {}
		answer['image'] = image_id
		answer['answer'] = reverse_adict[predict_label]
		predict_answers.append(answer)

	
	# save predict result
	out_file = options['out_dir'] + '/%s.json' % split
	logging.info(Fore.GREEN + 'saving the predict results to %s' %(out_file))
	with open(out_file,'w') as fid:
		json.dump(predict_answers,fid)
	
	if options['save_prob'] == 1:
		out_file = options['out_dir'] + '/%s_prob.pkl' % split
		pkl.dump(prob, open(out_file,'w'))

	coord.request_stop()
	coord.join(threads)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	options = evaluate_options()
	for key, value in options.items():
		parser.add_argument('--%s'%key, dest=key, type=type(value), default=None)
	args = parser.parse_args()
	args = vars(args)
	for key, value in args.items():
		if value:
			options[key] = value

	model_status_path = options['ckpt_prefix'] + '/status.json'
	model_status = json.load(open(model_status_path))
	del model_status['options']['ckpt_prefix']
	options.update(model_status['options'])

	options['out_dir'] = options['ckpt_prefix'] + '/results'
	if args['out_dir']:
		options['out_dir'] = args['out_dir']

	print(options)
	set_logging(options)
	os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu'])
	if not os.path.exists(options['out_dir']):
		os.makedirs(options['out_dir'])
	
	evaluate(options)


