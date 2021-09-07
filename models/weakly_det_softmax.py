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


class WeaklyDetSoftmax(model_base.ModelBase):
  """WeaklyDetSoftmax model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the WeaklyDetSoftmax instance. """
    if not isinstance(options, model_pb2.WeaklyDetSoftmax):
      raise ValueError('Options has to be an WeaklyDetSoftmax proto.')
    super(WeaklyDetSoftmax, self).__init__(options, is_training)

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

  def _id_to_one_hot(self, class_ids, n_classes):
    """Converts ids to one-hot representations. 

    NOTE: each value in class_ids are in [0, n_classes]. 0 denotes both background
          and padding.

    Args:
      class_ids: Class ids, [batch, max_n_action] int tensor.
      n_classes: Number of classes.

    Returns:
      labels: Label tensor using one-hot representation, [batch, 1 + n_classes].
    """
    one_hot = tf.one_hot(class_ids, depth=1 + n_classes, dtype=tf.float32)
    return tf.reduce_max(one_hot, axis=1)

  def _compute_class_aware_attention(self,
                                     video_features,
                                     video_lengths,
                                     n_classes=None,
                                     attention_dropout_rate=0.0,
                                     is_training=False,
                                     name=None):
    """Computes the attention distribution given the labels.

    Args:
      video_features: Video features, [batch, max_video_length, dims] tensor.
      video_lengths: Observation window, [batch] tensor.
      n_classes: Number of classes.
      attention_dropout_rate: Dropout rate of the attention layer.
      is_training: If true, apply dropout.
      name: Name of the attention layer.

    Returns:
      attention: A [batch, n_classes, max_video_length] tensor.
      pre_attention: A [batch, n_classes, max_video_length] tensor.
    """
    dims = video_features.shape[-1].value
    max_video_length = tf.shape(video_features)[1]

    # Compute attention masks.
    #   masks shape = [batch, 1, max_max_video_length].
    masks = tf.sequence_mask(
        video_lengths, maxlen=max_video_length, dtype=tf.float32)
    masks = tf.expand_dims(masks, 1)

    if is_training:
      random_masks = tf.greater(
          tf.random.uniform(tf.shape(masks), minval=0, maxval=1.0), attention_dropout_rate)
      masks = tf.multiply(masks, tf.cast(random_masks, tf.float32))

    # Compute attention.
    #   pre_attention shape = [batch, n_classes, max_max_video_length].
    #   attention shape = [batch, n_classes, max_max_video_length].

    attention_fc = tf.keras.layers.Dense(
        n_classes, 
        kernel_initializer=self._kernel_initializer, 
        kernel_regularizer=self._kernel_regularizer,
        name=name)
    self._layers.append(attention_fc)

    pre_attention = attention_fc(video_features)
    pre_attention = tf.transpose(pre_attention, [0, 2, 1])

    attention = masked_ops.masked_softmax(pre_attention, masks, dim=2)

    tf.summary.histogram('attention/%s/attentions' % name, attention)
    tf.summary.histogram('attention/%s/pre_attention' % name, pre_attention)

    return attention, pre_attention

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    options = self.options

    video_ids, video_features, audio_embeddings, gyroscope_features, accelerator_features, video_lengths = inputs
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

    verb_attention, _ = self._compute_class_aware_attention(
        video_features, 
        video_lengths, 
        n_classes=options.n_verb_classes,
        attention_dropout_rate=options.attention_dropout_rate,
        is_training=self.is_training,
        name='VerbAttention')
    noun_attention, _ = self._compute_class_aware_attention(
        video_features, 
        video_lengths, 
        n_classes=options.n_noun_classes,
        attention_dropout_rate=options.attention_dropout_rate,
        is_training=self.is_training,
        name='NounAttention')

    # Apply attention.
    #   verb_feature shape = [batch, n_verb_classes, dims].
    #   noun_feature shape = [batch, n_noun_classes, dims].
    verb_feature = tf.einsum('bcl,bld->bcd', verb_attention, video_features)
    noun_feature = tf.einsum('bcl,bld->bcd', noun_attention, video_features)

    # Action classification.
    #   verb_seq_classification shape = [batch, max_video_length, 1 + n_verb_classes]
    #   noun_seq_classification shape = [batch, max_video_length, 1 + n_noun_classes]
    verb_fc = tf.keras.layers.Dense(
        options.n_verb_classes, 
        kernel_initializer=self._kernel_initializer, 
        kernel_regularizer=self._kernel_regularizer,
        name='VerbFC')
    noun_fc = tf.keras.layers.Dense(
        options.n_noun_classes, 
        kernel_initializer=self._kernel_initializer, 
        kernel_regularizer=self._kernel_regularizer,
        name='NounFC')
    self._layers.extend([verb_fc, noun_fc])

    verb_logits = verb_fc(verb_feature)
    noun_logits = noun_fc(noun_feature)

    verb_seq_classification = verb_fc(video_features)
    noun_seq_classification = noun_fc(video_features)

    return {
        'video_id': video_ids,
        'video_features': video_features,
        'video_lengths': video_lengths,
        'verb_attention': verb_attention,
        'verb_seq_classification': verb_seq_classification,
        'verb_logits': verb_logits,
        'noun_attention': noun_attention,
        'noun_seq_classification': noun_seq_classification,
        'noun_logits': noun_logits,
    }

  def build_losses(self, predictions, labels, **kwargs):
    """Computes loss tensors.

    Args:
      predictions: A dictionary of prediction tensors keyed by name.
      labels: A dictionary of label tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    options = self.options

    # Classification loss.
    n_action, noun_ids, verb_ids = labels

    verb_logits = predictions['verb_logits']
    noun_logits = predictions['noun_logits']
    verb_labels = self._id_to_one_hot(verb_ids, options.n_verb_classes)
    noun_labels = self._id_to_one_hot(noun_ids, options.n_noun_classes)
    verb_labels, noun_labels = verb_labels[:, 1:], noun_labels[:, 1:]

    verb_grid_labels = tf.multiply(tf.expand_dims(verb_labels, 2), tf.expand_dims(verb_labels, 1))
    noun_grid_labels = tf.multiply(tf.expand_dims(noun_labels, 2), tf.expand_dims(noun_labels, 1))

    verb_losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=verb_grid_labels, logits=verb_logits)
    noun_losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=noun_grid_labels, logits=noun_logits)

    def reduce_mean(losses, masks):
      return tf.math.divide(tf.reduce_sum(losses * masks), 
                            1e-12 + tf.reduce_sum(masks))

    loss_dict = {
        'verb_loss': reduce_mean(verb_losses, verb_labels),
        'noun_loss': reduce_mean(noun_losses, noun_labels),
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
    verb_seq_classification = predictions['verb_seq_classification']
    verb_attention = predictions['verb_attention']

    #######################################################
    # Detection evaluation.
    #######################################################

    # Post-processing.
    verb_scores = tf.nn.softmax(verb_seq_classification)
    #verb_scores = tf.multiply(verb_scores, tf.transpose(verb_attention, [0, 2, 1]))  # TODO: check
    #verb_scores = tf.transpose(verb_attention, [0, 2, 1])  # TODO: check

    # (n_detection, i_start, i_end, verb_id, verb_score) = post_process(
    #     verb_scores[:, :, 1:],
    #     max_n_detection=1000,
    #     score_min=0.05,
    #     score_max=0.25,
    #     score_step=0.05)
    (n_detection, i_start, i_end, verb_id, verb_score) = post_process(
        verb_scores,
        max_n_detection=200,
        score_min=0.05,
        score_max=0.25,
        score_step=0.05)

    # Compute timestamp.
    video_id = predictions['video_id']
    video_lengths = predictions['video_lengths']
    video_features = predictions['video_features']
    duration_per_segment = tf.math.divide(
        self._id2length.lookup(video_id), 1e-12 + tf.cast(video_lengths, tf.float32))

    # Launch evaluator.
    metrics = self._evaluator.get_estimator_eval_metric_ops(eval_dict={
        'video_id': video_id,
        'n_detection': n_detection,
        't_start': tf.cast(i_start, tf.float32) * duration_per_segment,
        't_end': tf.cast(i_end + 1, tf.float32) * duration_per_segment,
        'verb_id': verb_id,
        'verb_score': verb_score,
    })
    metrics.update({
        'metrics/accuracy': metrics['metrics/verb_map_avg'],
    })
    return metrics
