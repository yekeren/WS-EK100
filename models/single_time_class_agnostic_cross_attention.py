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
#from stn import spatial_transformer_network as transformer

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = logits.shape[-1].value
    y_hard = tf.cast(tf.one_hot(tf.argmax(y, -1), k), y.dtype)
    #y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

def load_video_lengths(file_name):
  df = pd.read_csv(file_name, sep=',')
  return {i: l for i, l in zip(df['video_id'], df['duration'])}


class SingleTimeClassAgnosticCrossAttentionDet(model_base.ModelBase):
  """SingleTimeClassAgnosticCrossAttentionDet model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the SingleTimeClassAgnosticCrossAttentionDet instance. """
    if not isinstance(options, model_pb2.SingleTimeClassAgnosticCrossAttentionDet):
      raise ValueError('Options has to be an SingleTimeClassAgnosticCrossAttentionDet proto.')
    super(SingleTimeClassAgnosticCrossAttentionDet, self).__init__(options, is_training)

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
          activation=tf.nn.relu,
          use_bias=True,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          name='%s/Conv1d_Layer%i' % (scope, 1 + layer_id))
      video_features = conv1d(video_features)
      self._layers.append(conv1d)

      # Dropout.
      if self.is_training and options.conv1d.dropout_rate > 0:
        video_features = tf.keras.layers.Dropout(
            options.conv1d.dropout_rate)(video_features)
    return video_features


  def _config_features(self, 
                       rgb_features, 
                       flow_features, 
                       audio_features, 
                       gyroscope_features, 
                       accelerator_features, 
                       feature_config):
    features = []
    for flag, feature in zip(
        [feature_config.use_rgb, 
         feature_config.use_flow, 
         feature_config.use_audio, 
         feature_config.use_gyroscope, 
         feature_config.use_accelerator], 
        [rgb_features, 
         flow_features, 
         audio_features, 
         gyroscope_features, 
         accelerator_features]):
      if flag:
        features.append(feature)
      
    return tf.concat(features, -1)

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

    # Fusion.
    video_seq_features = self._config_features(rgb_features, 
                                               flow_features, 
                                               audio_features, 
                                               gyroscope_features, 
                                               accelerator_features, 
                                               options.feature_config)
    query_seq_features = self._config_features(rgb_features, 
                                               flow_features, 
                                               audio_features, 
                                               gyroscope_features, 
                                               accelerator_features, 
                                               options.query_config)
      
    video_seq_features = self._compute_video_features(video_seq_features, 'VideoSeqFeature')
    query_seq_features = self._compute_video_features(query_seq_features, 'QuerySeqFeature')

    # Compute attention.
    attention = tf.einsum('bld,bld->bl', video_seq_features, query_seq_features)
    verb_attention = noun_attention = attention

    # Apply attention.
    #   verb_features shape = [batch, n_verb_classes, dims].
    #   noun_features shape = [batch, n_noun_classes, dims].
    verb_features = tf.einsum('bl,bld->bd', verb_attention, video_seq_features)
    noun_features = tf.einsum('bl,bld->bd', noun_attention, video_seq_features)

    verb_features = tf.math.divide(verb_features, 1e-10 + tf.reduce_sum(verb_attention, -1, keepdims=True))
    noun_features = tf.math.divide(noun_features, 1e-10 + tf.reduce_sum(noun_attention, -1, keepdims=True))

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

    self._layers.extend([noun_fc, verb_fc])

    # Action classification.
    #   verb_seq_classification shape = [batch, max_video_length, 1 + n_verb_classes]
    #   noun_seq_classification shape = [batch, max_video_length, 1 + n_noun_classes]
    verb_seq_classification = verb_fc(video_seq_features)
    noun_seq_classification = noun_fc(video_seq_features)

    verb_logits = verb_fc(verb_features)
    noun_logits = noun_fc(noun_features)

    return {
        'video_id': video_ids,
        'video_lengths': video_lengths,
        'video_features': video_features,
        'video_seq_features': video_seq_features,
        'verb_logits': verb_logits,
        'verb_seq_attention': verb_attention,
        'verb_seq_classification': verb_seq_classification,
        'noun_logits': noun_logits,
        'noun_seq_attention': noun_attention,
        'noun_seq_classification': noun_seq_classification,
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

    noun_ids, verb_ids = labels
    noun_logits, verb_logits = predictions['noun_logits'], predictions['verb_logits']

    verb_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=verb_ids, logits=verb_logits)
    noun_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=noun_ids, logits=noun_logits)

    loss_dict = {
        'noun_loss': tf.reduce_mean(noun_losses),
        'verb_loss': tf.reduce_mean(verb_losses)
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
    verb_seq_attention = predictions['verb_seq_attention']

    #######################################################
    # Detection evaluation.
    #######################################################

    # Post-processing.
    verb_scores = tf.nn.softmax(verb_seq_classification, -1)

    (n_detection, i_start, i_end, verb_id, verb_score) = post_process(
        verb_scores[:, :, 1:],
        max_n_detection=1000,
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
