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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from protos import model_pb2
from models import model_base
from models.detection_evaluation import DetectionEvaluator
from models.post_process import post_process
from modeling.utils import masked_ops


def load_video_lengths(file_name):
  df = pd.read_csv(file_name, sep=',')
  return {i: l for i, l in zip(df['video_id'], df['duration'])}


class WSDet(model_base.ModelBase):
  """WSDet model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the WSDet instance. """
    if not isinstance(options, model_pb2.WSDet):
      raise ValueError('Options has to be an WSDet proto.')
    super(WSDet, self).__init__(options, is_training)

    self._evaluator = DetectionEvaluator(options.eval_annotation_path)

    # Video-id to length.
    id2length = load_video_lengths(options.video_info_csv_path)
    id2length = [(k, v) for k, v in id2length.items()]
    self._id2length = tf.lookup.StaticHashTable( 
        tf.lookup.KeyValueTensorInitializer(
          [x[0] for x in id2length], 
          [x[1] for x in id2length]), -1)

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    options = self.options

    video_ids, video_features, observation_windows = inputs

    #kernel_initializer='glorot_uniform'
    kernel_initializer='glorot_normal'

    if options.HasField('conv1d'):
      for layer_id in range(options.conv1d.layers):
        video_features = tf.keras.layers.Conv1D(
            options.conv1d.filters,
            options.conv1d.kernel_size,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            name='conv1d_%i' % (1 + layer_id))(video_features)
        if self.is_training and options.conv1d.dropout_rate > 0:
          video_features = tf.keras.layers.Dropout(
              options.conv1d.dropout_rate)(video_features)

    noun_logits = tf.keras.layers.Dense(
        1 + options.n_noun_classes,
        kernel_initializer=kernel_initializer)(video_features)
    verb_logits = tf.keras.layers.Dense(
        1 + options.n_verb_classes,
        kernel_initializer=kernel_initializer)(video_features)

    return {
        'video_id': video_ids,
        'observation_windows': observation_windows,
        'noun_logits': noun_logits,
        'verb_logits': verb_logits}

  def build_losses(self, predictions, labels, **kwargs):
    """Computes loss tensors.

    Args:
      predictions: A dictionary of prediction tensors keyed by name.
      labels: A dictionary of label tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    options = self.options

    import pdb
    pdb.set_trace()

    # Extract predictions and annotations.
    noun_labels, verb_labels = labels
    noun_logits, verb_logits = predictions['noun_logits'], predictions['verb_logits']

    # Compute losses.
    noun_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=noun_labels, logits=noun_logits)
    verb_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=verb_labels, logits=verb_logits)

    # Compute the loss masked mean.
    max_observation_window = tf.shape(verb_logits)[1]
    observation_windows = predictions['observation_windows']
    masks = tf.sequence_mask(observation_windows, max_observation_window, dtype=tf.float32)

    def masked_mean(losses):
      return tf.math.divide(tf.reduce_sum(tf.multiply(losses, masks)), 
                            1e-8 + tf.reduce_sum(masks))

    return {
        'noun_loss': masked_mean(noun_losses),
        'verb_loss': masked_mean(verb_losses)}

  def build_metrics(self, predictions, labels, **kwargs):
    """Computes evaluation metrics.

    Args:
      predictions: A dictionary of prediction tensors keyed by name.
      labels: A dictionary of label tensors keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    options = self.options

    # Extract predictions and annotations.
    noun_labels, verb_labels = labels
    noun_logits, verb_logits = predictions['noun_logits'], predictions['verb_logits']

    # Post-processing.
    verb_scores = tf.nn.softmax(verb_logits, -1)
    noun_scores = tf.nn.softmax(noun_logits, -1)
    (n_detection, i_start, i_end, verb_id, verb_score) = post_process(
        verb_scores[:, :, 1:],
        max_n_detection=1000,
        score_min=0.05,
        score_max=0.25,
        score_step=0.05)

    # Compute timestamp.
    observation_windows = predictions['observation_windows']
    video_id = predictions['video_id']
    duration_per_segment = tf.math.divide(
        self._id2length.lookup(video_id), 
        tf.cast(observation_windows, tf.float32))

    # Launch evaluator.
    metrics = self._evaluator.get_estimator_eval_metric_ops(eval_dict={
      'video_id': video_id,
      'n_detection': n_detection,
      't_start': tf.cast(i_start, tf.float32) * duration_per_segment,
      't_end': tf.cast(i_end + 1, tf.float32) * duration_per_segment,
      'verb_id': verb_id,
      'verb_score': verb_score,
      })
    metrics = {('metrics/' + k): v for k, v in metrics.items()}
    metrics.update({'metrics/accuracy': metrics['metrics/verb_map_avg']})
    return metrics
