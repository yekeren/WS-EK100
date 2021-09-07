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


class FullyDet(model_base.ModelBase):
  """FullyDet model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the FullyDet instance. """
    if not isinstance(options, model_pb2.FullyDet):
      raise ValueError('Options has to be an FullyDet proto.')
    super(FullyDet, self).__init__(options, is_training)

    self._evaluator = DetectionEvaluator(options.eval_annotation_path,
                                         options.noun_classes_csv,
                                         options.verb_classes_csv)

    # Video-id to length.
    id2length = load_video_lengths(options.video_info_csv_path)
    id2length = [(k, v) for k, v in id2length.items()]
    self._id2length = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            [x[0] for x in id2length],
            [x[1] for x in id2length]), -1)

    self._kernel_initializer = 'glorot_normal'
    self._kernel_regularizer = tf.keras.regularizers.l2(options.weight_decay)
    self._layers = []

  def _compute_video_features(self, video_features, scope):
    """Computes the video features.

    Args:
      video_features: A [batch, max_observation_window, feature_dims] tensor.
      scope: Variable scope.

    Returns:
      video_features: A [batch, max_observation_window, output_dims] tensor.
    """
    options = self.options

    for layer_id in range(options.conv1d.layers):
      # Conv1D
      conv1d = tf.keras.layers.Conv1D(
          options.conv1d.filters,
          options.conv1d.kernel_size,
          padding="same",
          activation=None,
          use_bias=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          name='%s/Conv1d_Layer%i' % (scope, 1 + layer_id))
      video_features = conv1d(video_features)
      self._layers.append(conv1d)

      # BatchNormalization.
      batch_norm = tf.keras.layers.BatchNormalization(
          scale=False,
          name='%s/BatchNorm_Layer%i' % (scope, 1 + layer_id))
      video_features = batch_norm(video_features, training=self.is_training)

      activation = tf.keras.layers.Activation('relu')
      video_features = activation(video_features)

      # Dropout.
      if self.is_training and options.conv1d.dropout_rate > 0:
        video_features = tf.keras.layers.Dropout(
            options.conv1d.dropout_rate)(video_features)
    return video_features

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    options = self.options

    video_ids, video_features, audio_embeddings, gyroscope_features, accelerator_features, observation_windows = inputs
    audio_features = tf.nn.relu(audio_embeddings)

    rgb_features, flow_features = tf.split(video_features, 2, axis=-1)

    video_features = []
    for scope, flag, feature in zip(
        ['RGB', 'Flow', 'Audio', 'Gyroscope', 'Accelerator' ],
        [options.feature_config.use_rgb, options.feature_config.use_flow, options.feature_config.use_audio, options.feature_config.use_gyroscope, options.feature_config.use_accelerator], 
        [rgb_features, flow_features, audio_features, gyroscope_features, accelerator_features]):
      if flag:
        video_features.append(self._compute_video_features(feature, scope))
      
    video_features = tf.concat(video_features, -1)

    noun_fc = tf.keras.layers.Dense(
        1 + options.n_noun_classes,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        name='NounFC')
    verb_fc = tf.keras.layers.Dense(
        1 + options.n_verb_classes,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        name='VerbFC')

    self._layers.extend([verb_fc, noun_fc])
    verb_logits = verb_fc(video_features)
    noun_logits = noun_fc(video_features)

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
    masks = tf.sequence_mask(
        observation_windows, max_observation_window, dtype=tf.float32)

    def masked_mean(losses):
      return tf.math.divide(tf.reduce_sum(tf.multiply(losses, masks)),
                            1e-8 + tf.reduce_sum(masks))

    loss_dict = {
        'noun_loss': masked_mean(noun_losses),
        'verb_loss': masked_mean(verb_losses)
    }

    # Regularization loss.
    for layer in self._layers:
      name = 'Regularization_' + layer.name
      loss_dict[name] = layer.losses[0]
    return loss_dict


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
    noun_seq_classification, verb_seq_classification = predictions['noun_logits'],predictions['verb_logits']

    #######################################################
    # Detection evaluation.
    #######################################################

    video_id = predictions['video_id']
    video_lengths = predictions['observation_windows']
    duration_per_segment = tf.math.divide(
        self._id2length.lookup(video_id), 1e-12 + tf.cast(video_lengths, tf.float32))

    # Post-processing.
    verb_scores = tf.nn.softmax(verb_seq_classification, -1)
    verb_scores = tf.expand_dims(verb_scores[:, :, 1:], 3)
    verb_scores = tf.tile(verb_scores, [1, 1, 1, options.n_noun_classes])
    verb_scores = tf.reshape(verb_scores, [1, -1, options.n_verb_classes * options.n_noun_classes])

    noun_scores = tf.nn.softmax(noun_seq_classification, -1)
    noun_scores = tf.expand_dims(noun_scores[:, :, 1:], 2)
    noun_scores = tf.tile(noun_scores, [1, 1, options.n_verb_classes, 1])
    noun_scores = tf.reshape(noun_scores, [1, -1, options.n_verb_classes * options.n_noun_classes])

    action_scores = tf.multiply(verb_scores, noun_scores)

    # Launch evaluator for action detection.
    (n_detection, i_start, i_end, action_id, action_score
     ) = post_process(action_scores,
                      max_n_detection=1000,
                      thresholds=[x ** 2 for x in [0.01, 0.1, 0.2, 0.4]])
    verb_id = action_id // options.n_noun_classes
    noun_id = action_id % options.n_noun_classes
    metrics = self._evaluator.get_estimator_eval_metric_ops(eval_dict={
        'video_id': video_id,
        'n_detection': n_detection,
        't_start': tf.cast(i_start, tf.float32) * duration_per_segment,
        't_end': tf.cast(i_end + 1, tf.float32) * duration_per_segment,
        'action_id': action_id,
        'action_score': action_score,
        'verb_id': verb_id,
        'noun_id': noun_id,
    })
    metrics.update({
        'metrics/accuracy': metrics['metrics/action_map_avg'],
    })

    return metrics
