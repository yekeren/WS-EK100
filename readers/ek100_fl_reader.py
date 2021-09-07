# Copyright 2020 Keren Ye, University of Pittsburgh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reads from tfrecord files and yields batched tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import os
import random
import scipy.interpolate
import numpy as np
import pandas as pd
import tensorflow as tf

from protos import reader_pb2


_MINUTES_TO_SECONDS = 60
_HOURS_TO_SECONDS = 60 * 60


random.seed(286)


def timestamp_to_seconds(timestamp):
  """Converts timestamp to seconds. """
  hours, minutes, seconds = map(float, timestamp.split(":"))
  total_seconds = hours * _HOURS_TO_SECONDS + \
      minutes * _MINUTES_TO_SECONDS + seconds
  return total_seconds


def _read_annotations(path_to_annotations):
  """Reads annotations from files.

  Args:
    path_to_annotations: Path to the annotation pkl file.

  Returns:
    groundtruth_df: A pandas.DataFrame instance.
  """
  annotations = pd.read_pickle(path_to_annotations)

  # Fix the entris with NA narration_timestamp.
  annotations = annotations.groupby('video_id', as_index=False).apply(
      lambda x: x.sort_values(by='narration_id', ascending=True).fillna(method='ffill').fillna(method='bfill'))

  groundtruth_df = pd.DataFrame({
      'video_id': annotations['video_id'],
      't_narration': annotations['narration_timestamp'].apply(timestamp_to_seconds),
      't_start': annotations['start_timestamp'].apply(timestamp_to_seconds),
      't_end': annotations['stop_timestamp'].apply(timestamp_to_seconds),
      'verb_class': annotations['verb_class'],
      'noun_class': annotations['noun_class'],
  })
  return groundtruth_df


def _read_video_features(video_ids, feature_directory):
  """Reads video features.

  Args:
    video_ids: Video ids, a [n_video_ids] ndarray.
    feature_directory: Path to the directory storing video features.

  Returns:
    observation_windows: A [n_video_ids] ndarray.
    video_features: A list of [observation_window, 2048] ndarray.
  """
  features = []
  for video_id in video_ids:
    feature = np.load(os.path.join( feature_directory, '{}.npy'.format(video_id)))
    features.append(feature.astype(np.float32).transpose([1, 0]))
  observation_windows = [x.shape[0] for x in features]
  return np.array(observation_windows, np.int32), features


def _read_lengths(video_ids, path_to_video_lengths):
  """Reads video lengths.

  Args:
    video_ids: Video ids, a [n_video_ids] ndarray.
    path_to_video_lengths: Path to the csv file storing video lengths.

  Returns:
    video_lengths: Video lengths, a [n_video_ids] ndarray.
  """
  annotations = pd.read_csv(path_to_video_lengths, sep=',')
  id2length = {}
  for _, row in annotations.iterrows():
    id2length[row['video_id']] = row['duration']

  lengths = []
  for video_id in video_ids:
    lengths.append(id2length[video_id])
  return np.stack(lengths)


def _resize_feature(input_feature, new_size):
  """Resizes features to new size.

  Args:
    input_feature: A [original_size, dims] ndarray.
    new_size: Target size.

  Returns:
    A [new_size, dims] ndarray.
  """
  original_size = len(input_feature)
  x = np.array(range(original_size))
  f = scipy.interpolate.interp1d(x, input_feature, axis=0)
  x_new = [i * float(original_size - 1) / (new_size - 1) for i in range(new_size)]
  y_new = f(x_new)
  return y_new.astype(np.float32)


def _read_audio_data(video_ids, 
                     video_lengths, 
                     audio_data_directory, 
                     cache_directory='cache',
                     flag='-audif'):
  """Reads audio data.

  Args:
    video_ids: Video ids, a [n_video_ids] ndarray.
    video_lengths: Video lengths, a [n_video_ids] ndarray.
    audio_data_directory: Path to the directory storing audio features.
    cache_directory: Path to the directory storing cache files.

  Returns:
    audio_features: A list of [video_length, 128] ndarray.
  """
  audio_features = []
  for video_id, video_length in zip(video_ids, video_lengths):
    audio_cache_file_name = os.path.join(cache_directory, video_id + '{}.npy'.format(flag))

    if os.path.isfile(audio_cache_file_name):
      audio_features.append(np.load(audio_cache_file_name))
    else:
      logging.info('Processing audio data for %s...', video_id)

      audio_feature = np.load(os.path.join(audio_data_directory, video_id + '.npy'))
      audio_features.append(_resize_feature(audio_feature, video_length))
      np.save(audio_cache_file_name, audio_features[-1])
  return audio_features


def _read_meta_data(video_ids, 
                    video_lengths, 
                    meta_data_directory, 
                    cache_directory='cache'):
  """Reads gyroscope and accelerator data.

  Args:
    video_ids: Video ids, a [n_video_ids] ndarray.
    video_lengths: Video lengths, a [n_video_ids] ndarray.
    meta_data_directory: Path to the directory storing meta data.
    cache_directory: Path to the directory storing cache files.

  Returns:
    gyro_features: A list of [video_length, 3] ndarray.
    accl_features: A list of [video_length, 3] ndarray.
  """
  gyro_features = []
  accl_features = []

  for video_id, video_length in zip(video_ids, video_lengths):
    kichen_id = video_id.split('_')[0]

    # Read gyro data.
    gyro_cache_file_name = os.path.join(cache_directory, video_id + '-gyro.npy')
    if os.path.isfile(gyro_cache_file_name):
      gyro_features.append(np.load(gyro_cache_file_name))
    else:
      logging.info('Processing gyroscope data for %s...', video_id)
      gyro_file_name = os.path.join(
          meta_data_directory, kichen_id, 'meta_data', video_id + '-gyro.csv')
      if os.path.isfile(gyro_file_name):
        gyro_df = pd.read_csv(gyro_file_name).filter(items=['GyroX', 'GyroY', 'GyroZ'])
        gyro_features.append(_resize_feature(gyro_df.values, video_length))
      else:
        gyro_features.append(np.zeros((video_length, 3), dtype=np.float32))
      np.save(gyro_cache_file_name, gyro_features[-1])

    # Read accl data.
    accl_cache_file_name = os.path.join(cache_directory, video_id + '-accl.npy')
    if os.path.isfile(accl_cache_file_name):
      accl_features.append(np.load(accl_cache_file_name))
    else:
      logging.info('Processing accelerator data for %s...', video_id)
      accl_file_name = os.path.join(
          meta_data_directory, kichen_id, 'meta_data', video_id + '-accl.csv')
      if os.path.isfile(accl_file_name):
        accl_df = pd.read_csv(accl_file_name).filter(items=['AcclX', 'AcclY', 'AcclZ'])
        accl_features.append(_resize_feature(accl_df.values, video_length))
      else:
        accl_features.append(np.zeros((video_length, 3), dtype=np.float32))
      np.save(accl_cache_file_name, accl_features[-1])

  return gyro_features, accl_features

def _parse_instance_labels(groundtruth_df, video_ids, video_lengths, observation_windows):
  """Parses the instance-level labels for the fully-supervised methods.

  Args:
    groundtruth_df: A pandas.DataFrame instance.
    video_ids: Video ids, a [n_video_ids] ndarray.
    video_lengths: Video lengths, a [n_video_ids] ndarray.
    observation_windows: Length of video features, a [n_video_ids] ndarray.

  Returns:
    noun_labels: Instance-level noun labels, a list of [observation_windows] ndarray.
    verb_labels: Instance-level verb labels, a list of [observation_windows] ndarray.
  """
  noun_labels, verb_labels = [], []
  for observation_window in observation_windows:
    verb_labels.append(np.full((observation_window,), -1, dtype=np.int32))
    noun_labels.append(np.full((observation_window,), -1, dtype=np.int32))


  videoid2index = {video_id: i for i, video_id in enumerate(video_ids)}
  for _, row in groundtruth_df.iterrows():
    index = videoid2index[row['video_id']]
    duration = video_lengths[index]
    observation_window = observation_windows[index]

    # Find the associate bins in range of [0, observation_window).
    bin_start = int(observation_window * row['t_start'] / duration)
    bin_end = int(observation_window * row['t_end'] / duration)
    for bin_index in range(bin_start, min(bin_end + 1, observation_window)):
      noun_labels[index][bin_index] = row['noun_class']
      verb_labels[index][bin_index] = row['verb_class']

  # Offset the labels to start with 1, then 0 to denote background.
  for index in range(len(video_ids)):
    noun_labels[index] += 1
    verb_labels[index] += 1
  return noun_labels, verb_labels


def _parse_instance_labels(groundtruth_df, video_ids, video_lengths, observation_windows):
  """Parses the instance-level labels for the fully-supervised methods.

  Args:
    groundtruth_df: A pandas.DataFrame instance.
    video_ids: Video ids, a [n_video_ids] ndarray.
    video_lengths: Video lengths, a [n_video_ids] ndarray.
    observation_windows: Length of video features, a [n_video_ids] ndarray.

  Returns:
    ws_t_start: clip starting time, a list of [n_action] ndarray.
    ws_t_end: clip ending time, a list of [n_action] ndarray.
    ws_noun_labels: clip-level noun labels, a list of [n_action] ndarray.
    ws_verb_labels: clip-level verb labels, a list of [n_action] ndarray.
  """
  ws_t_start = []
  ws_t_end = []
  ws_verb_label, ws_noun_label = [], []

  for video_id, video_length in zip(video_ids, video_lengths):
    df = groundtruth_df[groundtruth_df.video_id == video_id].sort_values(by='t_narration')

    ws_t_start.append(df.t_start.to_numpy())
    ws_t_end.append(df.t_end.to_numpy())
    ws_verb_label.append(1 + df.verb_class.to_numpy())
    ws_noun_label.append(1 + df.noun_class.to_numpy())

  return ws_t_start, ws_t_end, ws_noun_label, ws_verb_label


def _bisearch_left(t_narration, value):
  """Processes binary search to seek the start index.

  Args:
    t_narration: Narration array.
    value: Starting time.

  Returns:
    Index i, such that t_narration[i] <= value < t_narration[i + 1].
  """
  if value <= t_narration[0]: return 0

  l = len(t_narration)
  left, right = 0, l - 1
  while left < right:
    mid = (left + right + 1) // 2
    if value == t_narration[mid]:
      return mid
    elif value > t_narration[mid]:
      left = mid
    else:
      right = mid - 1
  return left

def _bisearch_right(t_narration, value):
  """Processes binary search to seek the end index.

  Args:
    t_narration: Narration array.
    value: Starting time.

  Returns:
    Index i, such that t_narration[i - 1] < value <= t_narration[i].
  """
  l = len(t_narration)
  if value >= t_narration[-1]: return l

  left, right = 0, l - 1
  while left < right:
    mid = (left + right) // 2
    if value == t_narration[mid]:
      return mid
    elif value > t_narration[mid]:
      left = mid + 1
    else:
      right = mid
  return left

def _create_dataset(options, is_training, input_pipeline_context=None):
  """Creates dataset object based on options.

  Args:
    options: An instance of reader_pb2.Reader.
    is_training: If true, shuffle the dataset.
    input_pipeline_context: A tf.distribute.InputContext instance.

  Returns:
    A tf.data.Dataset object.
  """
  groundtruth_df = _read_annotations(options.path_to_annotations)
  video_ids = pd.unique(groundtruth_df['video_id'])

  # Read video lengths, shape = [n_video_ids].
  video_lengths = _read_lengths(video_ids, options.path_to_video_lengths)

  # Load video features, a list of [observation_window, feature_dims] ndarray.
  observation_windows, video_features = _read_video_features(
      video_ids,
      options.video_feature_directory)

  # Read audio data.
  # audio_features = _read_audio_data(
  #     video_ids, 
  #     observation_windows, 
  #     options.audio_feature_directory,
  #     cache_directory=options.cache_directory,
  #     flag='-audif')
  audio_embeddings = _read_audio_data(
      video_ids, 
      observation_windows, 
      options.audio_embedding_directory,
      cache_directory=options.cache_directory,
      flag='-audie')

  # Read gyroscope and accelerator data.
  gyroscope_features, accelerator_features = _read_meta_data(
      video_ids, 
      observation_windows,
      options.meta_data_directory,
      cache_directory=options.cache_directory)

  # # Parse instance labels, shape = [n_video_ids, max_observation_window]..
  # noun_inst_labels, verb_inst_labels = _parse_instance_labels(
  #     groundtruth_df, video_ids, video_lengths, observation_windows)

  ws_t_starts, ws_t_ends, ws_noun_labels, ws_verb_labels = _parse_instance_labels(
      groundtruth_df, video_ids, video_lengths, observation_windows)

  # Create TF dataset.

  def data_generator():
    """Yields (video_id, video_feature, video_length), (n_action, noun_label, verb_label)."""
    for video_id, video_feature, audio_embedding, gyroscope_feature, accelerator_feature, video_length, observation_window, ws_t_start, ws_t_end, ws_noun_label, ws_verb_label in zip(
        video_ids, video_features, audio_embeddings, gyroscope_features, accelerator_features, video_lengths, observation_windows, ws_t_starts, ws_t_ends, ws_noun_labels, ws_verb_labels):

      if is_training:
        index = random.randint(0, len(ws_t_start) - 1)
        verb_label, noun_label = ws_verb_label[index], ws_noun_label[index]
        
        start = ws_t_start[index]
        end = ws_t_end[index]

        # Randomize offset.
        if options.random_offset_range > 0:
          rand_start = np.random.uniform(start - options.random_offset_range, start)
          rand_end = np.random.uniform(end, end + options.random_offset_range)
          rand_start = max(rand_start, 0)
          rand_end = min(rand_end, video_length)
          start, end = rand_start, rand_end

        start, end = int(start * observation_window / video_length), int(end * observation_window / video_length)
        end = max(start + 1, end)

      else:
        noun_label = verb_label = 0
        start, end = 0, observation_window

      yield ((video_id, video_feature[start:end, :], audio_embedding[start:end, :], gyroscope_feature[start:end, :], accelerator_feature[start:end, :], end - start), (noun_label, verb_label))


  dataset = tf.data.Dataset.from_generator(
      data_generator, 
      output_types=((tf.string, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32), (tf.int32, tf.int32)), 
      output_shapes=(
        ([], [None, 2048], [None, 128], [None, 3], [None, 3], []), ([], [])))
  if is_training:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=options.shuffle_buffer_size)
  dataset = dataset.padded_batch(options.batch_size,
      padded_shapes=(([], [None, 2048], [None, 128], [None, 3], [None, 3], []), ([], [])),
      drop_remainder=True)
  dataset = dataset.prefetch(options.prefetch_buffer_size)
  return dataset


def get_input_fn(options, is_training):
  """Returns a function that generate input examples.

  Args:
    options: An instance of reader_pb2.Reader.
    is_training: If true, shuffle the dataset.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.EK100FLReader):
    raise ValueError(
        'options has to be an instance of SceneGraphTextGraphReader.')

  def _input_fn(input_pipeline_context=None):
    """Returns a python dictionary.

    Returns:
      A dataset that can be fed to estimator.
    """
    return _create_dataset(options, is_training, input_pipeline_context)

  return _input_fn
