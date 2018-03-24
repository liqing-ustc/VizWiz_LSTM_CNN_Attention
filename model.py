import tensorflow as tf
import math
import numpy as np
import json
import os
from utils import *
import sys
from inputs import *

class VQAModel(object):
	def __init__(self, options, mode='train'):
		self.initializer = tf.random_uniform_initializer(
			minval = - options['init_scale'],
			maxval = options['init_scale']
		)
		self.regularizer = tf.contrib.layers.l2_regularizer(1.0)
		self.reader = tf.TFRecordReader()

		assert mode in ['train', 'eval', 'predict']
		if mode == 'train':
			self.is_training = True
		else:
			self.is_training = False
		self.mode = mode

		
		options['values_per_input_shard'] = 1000
		options['input_queue_capacity_factor'] = 10
		options['num_input_reader_threads'] = 4
		options['num_preprocess_threads'] = 4 
		self.options = options
		
	
	def build_inputs(self):
		options = self.options
		input_queue = prefetch_input_data(
			self.reader,
			options['input_file_pattern'],
			is_training=self.is_training,
			batch_size=options['batch_size'],
			values_per_shard=options['values_per_input_shard'],
			input_queue_capacity_factor=options['input_queue_capacity_factor'],
			num_reader_threads=options['num_input_reader_threads']
		)

		examples = []
		for thread_id in range(options['num_preprocess_threads']):
			serialized_example = input_queue.dequeue()
			single_example = parse_example(serialized_example)
  			feat_map = tf.SparseTensor(
				  indices=single_example['feat_map_idx'],
				  values=single_example['feat_map_data'],
				  dense_shape=[options['num_region'] * options['img_feat_dim']])
			feat_map = tf.sparse_tensor_to_dense(feat_map)
			feat_map = tf.reshape(feat_map, [options['img_feat_dim'], options['num_region']])
			feat_map = tf.transpose(feat_map, [1, 0])
			feat_map = tf.nn.l2_normalize(feat_map, dim=-1)
			del single_example['feat_map_idx']
			del single_example['feat_map_data']
			single_example['feat_map'] = feat_map
			single_example['question_mask'] = tf.ones_like(single_example['question'], dtype=tf.int32)
			single_example['answer'] = tf.maximum(single_example['answer'], 0)
			examples.append(single_example)

		# Batch inputs
		queue_capacity = (2 * options['num_preprocess_threads'] * options['batch_size'])
		inputs = batch_with_dynamic_pad(examples, batch_size=options['batch_size'], queue_capacity=queue_capacity)
		return inputs

	def build_question_embedding(self, question):
		question_emb = tf.contrib.layers.embed_sequence(
			question,
			vocab_size=self.options['vocab_size'],
			embed_dim=self.options['word_emb_size'],
			initializer=self.initializer,
			#regularizer=self.regularizer
		)
		return question_emb

	def build_question_rnn(self, question_emb, question_mask, keep_prob):
		rnn_cell = tf.contrib.rnn.LSTMCell(
			num_units=self.options['rnn_size'],
			state_is_tuple=True, 
			initializer=tf.orthogonal_initializer()
			)

		rnn_cell = tf.contrib.rnn.DropoutWrapper(
			rnn_cell,
			input_keep_prob=keep_prob,
			output_keep_prob=1,
			state_keep_prob=1
		)

		sequence_length = tf.reduce_sum(question_mask, 1)
		_, question_rnn_state = tf.nn.dynamic_rnn(cell=rnn_cell,
											inputs=question_emb,
											sequence_length=sequence_length,
											dtype=tf.float32
											)
		question_rnn_enc = getattr(question_rnn_state, 'h', question_rnn_state)
		return question_rnn_enc

	def build(self):
		options = self.options
		word_emb_size = self.options['word_emb_size']
		img_feat_dim = self.options['img_feat_dim']
		n_attention = self.options['attention_size']
		
		with tf.variable_scope('inputs'):
			inputs = self.build_inputs()
		outputs = {}
		outputs['image_id'] = inputs['image_id']


		if self.is_training:
			if self.options['drop_prob'] > 0:
				logging.debug(Fore.CYAN + 'using dropout!')
			drop_prob = tf.constant(self.options['drop_prob'], name='drop_prob')
		else:
			drop_prob = tf.constant(0., name='drop_prob')


		## question
		with tf.variable_scope('question'):
			question = inputs['question']
			question_mask = inputs['question_mask']

			with tf.variable_scope('embedding'):
				question_emb = self.build_question_embedding(question)
				question_emb = tf.nn.tanh(question_emb)
				tf.summary.histogram('question_emb', question_emb)

			with tf.variable_scope('rnn'):
				question_enc = self.build_question_rnn(
					question_emb,
					question_mask,
					keep_prob= 1 - drop_prob)
				tf.summary.histogram('question_enc', question_enc)

		## image 
		with tf.variable_scope('feat_map'):
			feat_map = inputs['feat_map']
			tf.summary.histogram('feat_map', feat_map)

		### attention
		with tf.variable_scope('attention'):
			question_enc_tile = tf.tile(tf.expand_dims(question_enc, 1), [1,self.options['num_region'],1])
			att_combines = tf.concat([feat_map, question_enc_tile], axis=-1)
			att_combines = tf.reshape(att_combines, shape=(-1, 14, 14, self.options['rnn_size']+self.options['img_feat_dim']))
			with tf.variable_scope('conv_1'):
				att_combines = tf.contrib.layers.conv2d(
					inputs=tf.nn.dropout(att_combines, keep_prob=1.0 - drop_prob),
					num_outputs=n_attention,
					kernel_size=1,
					stride=1,
					padding='SAME',
					data_format='NHWC',
					activation_fn=tf.nn.relu,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					weights_regularizer=self.regularizer,
					)

			with tf.variable_scope('conv_2'):
				att_combines = tf.contrib.layers.conv2d(
					inputs=tf.nn.dropout(att_combines, keep_prob=1.0 - drop_prob),
					num_outputs=1,
					kernel_size=1,
					stride=1,
					padding='SAME',
					data_format='NHWC',
					activation_fn=None,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					weights_regularizer=self.regularizer,
					)
				tf.summary.image('attention_map', att_combines)
			att_combines = tf.reshape(att_combines, (-1, self.options['num_region']))
			prob_attention = tf.nn.softmax(att_combines)
			outputs['prob_attention'] = prob_attention
			tf.summary.histogram('prob_attention ', prob_attention)
		
		with tf.variable_scope('image_feat'):
			img_feat = tf.einsum('ai,aij->aj', prob_attention, feat_map)
		
		## combines 
		with tf.variable_scope('combines'):
			combines = tf.concat([question_enc, img_feat], axis=-1)

		if self.options['two_final_fc'] == 1:
			with tf.variable_scope('final_fc') as scope:
				combines = tf.contrib.layers.fully_connected(
					inputs=tf.nn.dropout(combines, keep_prob=1.0 - drop_prob),
					num_outputs=self.options['final_fc_size'],
					activation_fn=tf.nn.relu,
					weights_initializer=tf.contrib.layers.xavier_initializer(),
					weights_regularizer=self.regularizer,
					scope=scope)

		with tf.variable_scope('logits_vizwiz') as scope:
			logits = tf.contrib.layers.fully_connected(
				inputs=tf.nn.dropout(combines, keep_prob=1.0 - drop_prob),
				num_outputs=self.options['n_answer_class'],
				activation_fn=None,
				weights_regularizer=self.regularizer,
				scope=scope)



		answer = inputs['answer']
		with tf.variable_scope('loss'):
			ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answer)
			ce_loss = tf.reduce_mean(ce_loss) / np.log(self.options['n_answer_class'])
			tf.summary.scalar('ce_loss', ce_loss)

		with tf.variable_scope('accuracy'):
			prob = tf.nn.softmax(logits)
			preds = tf.cast(tf.argmax(logits, 1), tf.int32)
			correct_preds = tf.equal(preds, answer)
			accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
			tf.summary.scalar('accuracy', accuracy)

			outputs.update({
				'prob': prob,
				'preds': preds,
				'accuracy': accuracy
				})

		with tf.variable_scope('regularization'):
			reg_loss = tf.losses.get_regularization_loss()
			tf.summary.scalar('reg_loss', reg_loss)

		with tf.variable_scope('final_loss'):
			loss = ce_loss + self.options['reg'] * reg_loss
			outputs['loss'] = loss

		return outputs