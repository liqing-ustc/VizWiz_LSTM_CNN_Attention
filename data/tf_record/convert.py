import sys
import os
import threading
from datetime import datetime
import h5py
import json
import numpy as np
import tensorflow as tf

batch_size = 1
num_threads = 8

def _int64_feature(value):
	"""Wrapper for inserting an int64 Feature into a SequenceExample proto."""
 	if type(value) != list:
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
	"""Wrapper for inserting an int64 Feature into a SequenceExample proto."""
	if type(value) != list:
		value = [value]
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _process_image_files(thread_index, ranges, split, dataset, fid_h5, num_shards):
	"""Processes and saves a subset of images as TFRecord files in one thread.

	Args:
	thread_index: Integer thread identifier within [0, len(ranges)].
	ranges: A list of pairs of integers specifying the ranges of the dataset to
		process in parallel.
	split: Unique identifier specifying the dataset.
	images: List of ImageMetadata.
	decoder: An ImageDecoder object.
	vocab: A Vocabulary object.
	num_shards: Integer number of shards for the output files.
	"""
	# Each thread produces N shards where N = num_shards / num_threads. For
	# instance, if num_shards = 128, and num_threads = 2, then the first thread
	# would produce shards [0, 64).
	num_threads = len(ranges)
	#assert not num_shards % num_threads
	num_shards_per_batch = int(num_shards / num_threads)

	shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
							 num_shards_per_batch + 1).astype(int)
	num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

	counter = 0
	for s in xrange(num_shards_per_batch):
		# Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
		shard = thread_index * num_shards_per_batch + s
		output_filename = "%.5d-of-%.5d" % (shard, num_shards)
		output_file = os.path.join(output_dir, output_filename)
		writer = tf.python_io.TFRecordWriter(output_file)

		shard_counter = 0
		images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
		for i in images_in_shard:
			image_id = dataset['image_id'][i]
			question = dataset['question'][i]
			answer = dataset['answer'][i]
			answerable = dataset['answerable'][i]
			image_feat_id = dataset['image_feat_id'][i]
			feat_map = fid_h5[image_feat_id][:].flatten()
			feat_map_data = feat_map[feat_map!=0].tolist()
			feat_map_idx = np.where(feat_map!=0)[0].tolist()
			

			example = tf.train.Example(
				features=tf.train.Features(
					feature={
						'image_id': _bytes_feature(image_id),
						'question': _int64_feature(question),
						'answer': _int64_feature(answer),
						'answerable': _int64_feature(answerable),
						'feat_map_idx': _int64_feature(feat_map_idx),
						'feat_map_data': _float_feature(feat_map_data)
					}
				)
			)

			if example is not None:
				writer.write(example.SerializeToString())
				shard_counter += 1
				counter += 1

			if not counter % 1000:
				print("%s [thread %d]: Processed %d of %d items in thread batch." %
						(datetime.now(), thread_index, counter, num_images_in_thread))
				sys.stdout.flush()

		writer.close()
		print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
				(datetime.now(), thread_index, shard_counter, output_file))
		sys.stdout.flush()
		shard_counter = 0
	print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
		(datetime.now(), thread_index, counter, num_shards_per_batch))
	sys.stdout.flush()


for split in ['val', 'train','test']:
	output_dir = split
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	dataset = json.load(open('../encode_QA/%s.json'%split))
	fid_h5 = h5py.File('../extract_feat/VizWiz.h5', 'r')['res5c']
	n_samples = len(dataset['image_id'])
	num_shards = int(np.ceil(float(n_samples) / 1000))
	print('n_samples: %d, num_shards: %d'%(n_samples, num_shards))

	# Break the images into num_threads batches. Batch i is defined as
	# images[ranges[i][0]:ranges[i][1]].
	num_threads = min(num_shards, num_threads)
	spacing = np.linspace(0, n_samples, num_threads + 1).astype(np.int)
	ranges = []
	threads = []
	for i in xrange(len(spacing) - 1):
		ranges.append([spacing[i], spacing[i + 1]])

	# Create a mechanism for monitoring when all threads are finished.
	coord = tf.train.Coordinator()

	# Launch a thread for each batch.
	print("Launching %d threads for spacings: %s" % (num_threads, ranges))
	for thread_index in xrange(len(ranges)):
		args = (thread_index, ranges, split, dataset, fid_h5, num_shards)
		t = threading.Thread(target=_process_image_files, args=args)
		t.start()
		threads.append(t)

	# Wait for all the threads to terminate.
	coord.join(threads)
	print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
		(datetime.now(), n_samples, split))
