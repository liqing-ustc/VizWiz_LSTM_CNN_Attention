from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def parse_example(serialized):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    caption_feature: Name of SequenceExample feature list containing integer
      captions.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    caption: A 1-D uint64 Tensor with dynamically specified length.
  """
  single_example = tf.parse_single_example(
      serialized,
      features={
          'image_id': tf.FixedLenFeature([], tf.string),
          'question': tf.VarLenFeature(tf.int64),
          'answer': tf.FixedLenFeature([], tf.int64),
          'feat_map_idx': tf.VarLenFeature(tf.int64),
          'feat_map_data': tf.VarLenFeature(tf.float32) #TODO: check if float32 ? or float64 ?
      }
    )
  single_example['image_id'] = tf.cast(single_example['image_id'], tf.string)
  single_example['question'] = tf.cast(tf.sparse_tensor_to_dense(single_example['question']), tf.int32)
  single_example['answer'] = tf.cast(single_example['answer'], tf.int32)
  single_example['feat_map_idx'] = tf.expand_dims(tf.sparse_tensor_to_dense(single_example['feat_map_idx']), axis=-1)
  single_example['feat_map_data'] = tf.sparse_tensor_to_dense(single_example['feat_map_data'])
  return single_example


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_capacity_factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.summary.scalar(
      "%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def batch_with_dynamic_pad(examples,
                           batch_size,
                           queue_capacity,
                           add_summaries=False):
  """Batches input images and captions.

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.

  Example:
    Actual captions in the batch ('-' denotes padded character):
      [
        [ 1 2 5 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    images: A Tensor of shape [batch_size, height, width, channels].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: An int32 Tensor of shape [batch_size, padded_length].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
  """

  inputs = tf.train.batch_join(
      examples,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      allow_smaller_final_batch=True,
      name="queue/batch_and_pad_queue")

  if add_summaries:
    lengths = tf.add(tf.reduce_sum(inputs['question_mask'], 1), 1)
    tf.summary.scalar("question_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("question_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("question_length/batch_mean", tf.reduce_mean(lengths))

  return inputs
