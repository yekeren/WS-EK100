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

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from protos import reader_pb2


_MINUTES_TO_SECONDS = 60
_HOURS_TO_SECONDS = 60 * 60


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
  groundtruth_df = pd.DataFrame({
      'video_id': annotations['video_id'],
      't_start': annotations['start_timestamp'].apply(timestamp_to_seconds),
      't_end': annotations['stop_timestamp'].apply(timestamp_to_seconds),
      'verb_class': annotations['verb_class'],
      'noun_class': annotations['noun_class'],
  })
  return groundtruth_df


def _read_features(video_ids, feature_directory):
  """Reads video features.

  Args:
    video_ids: Video ids, a [n_video_ids] ndarray.
    feature_directory: Path to the directory storing video features.

  Returns:
    observation_windows: A [n_video_ids] ndarray.
    video_features: A list of [observation_window, feature_dims] ndarray.
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

  # Load features, a list of [observation_window, feature_dims] features.
  observation_windows, video_features = _read_features(video_ids, options.feature_directory)
  feature_dims = video_features[0].shape[-1]

  # Read video lengths, shape = [n_video_ids].
  video_lengths = _read_lengths(video_ids, options.path_to_video_lengths)

  # Parse instance labels, shape = [n_video_ids, max_observation_window]..
  noun_labels, verb_labels = _parse_instance_labels(
      groundtruth_df, video_ids, video_lengths, observation_windows)


  # Create TF dataset.
  def data_generator():
    for video_id, video_feature, observation_window, noun_label, verb_label in zip(
        video_ids, video_features, observation_windows, noun_labels, verb_labels):
      yield ((video_id, video_feature, observation_window), (noun_label, verb_label))


  dataset = tf.data.Dataset.from_generator(
      data_generator, 
      output_types=((tf.string, tf.float32, tf.int32), (tf.int32, tf.int32)), 
      output_shapes=(
        (tf.TensorShape([]), tf.TensorShape([None, feature_dims]), tf.TensorShape([])), 
        (tf.TensorShape([None]), tf.TensorShape([None]))))
  if is_training:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=options.shuffle_buffer_size)
  dataset = dataset.padded_batch(options.batch_size,
      padded_shapes=(([], [None, feature_dims], []), ([None], [None])))
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
  if not isinstance(options, reader_pb2.EK100Reader):
    raise ValueError(
        'options has to be an instance of SceneGraphTextGraphReader.')

  def _input_fn(input_pipeline_context=None):
    """Returns a python dictionary.

    Returns:
      A dataset that can be fed to estimator.
    """
    return _create_dataset(options, is_training, input_pipeline_context)

  return _input_fn
