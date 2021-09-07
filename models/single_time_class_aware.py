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


class SingleTimeClassAwareDet(model_base.ModelBase):
  """SingleTimeClassAwareDet model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the SingleTimeClassAwareDet instance. """
    if not isinstance(options, model_pb2.SingleTimeClassAwareDet):
      raise ValueError('Options has to be an SingleTimeClassAwareDet proto.')
    super(SingleTimeClassAwareDet, self).__init__(options, is_training)

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


  def _compute_class_aware_attention(self,
                                     video_features,
                                     video_lengths,
                                     n_classes=None,
                                     use_gumbel_softmax=False,
                                     gumbel_softmax_temperature=0.5,
                                     attention_dropout_rate=0.0,
                                     sparse_regularizer=0.0,
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

    if not use_gumbel_softmax:
      attention_fc = tf.keras.layers.Dense(
          n_classes, 
          kernel_initializer=self._kernel_initializer, 
          kernel_regularizer=self._kernel_regularizer,
          name=name)
      self._layers.append(attention_fc)

      pre_attention = attention_fc(video_features)
      pre_attention = tf.transpose(pre_attention, [0, 2, 1])

      attention = tf.multiply(tf.nn.sigmoid(pre_attention), masks)
    else:
      attention_fc = tf.keras.layers.Dense(
          n_classes * 2, 
          kernel_initializer=self._kernel_initializer, 
          kernel_regularizer=self._kernel_regularizer,
          name=name)
      self._layers.append(attention_fc)

      batch = video_features.shape[0].value
      pre_attention = tf.reshape(attention_fc(video_features), [batch, -1, n_classes, 2])
      if self.is_training:
        attention = gumbel_softmax(pre_attention, temperature=gumbel_softmax_temperature)
      else:
        attention = tf.stop_gradient(tf.cast(tf.nn.softmax(pre_attention), tf.float32))

      attention = tf.transpose(attention[:, :, :, 0], [0, 2, 1])
      pre_attention = tf.transpose(pre_attention[:, :, :, 0], [0, 2, 1])
      attention = tf.multiply(masks, attention)

    tf.summary.histogram('attention/%s/attentions' % name, attention)
    tf.summary.histogram('attention/%s/pre_attention' % name, pre_attention)

    l1_loss = tf.div(tf.reduce_sum(attention * masks), n_classes * tf.reduce_sum(masks))
    tf.compat.v1.summary.scalar('losses/regularization/' + name, l1_loss)
    if sparse_regularizer > 0:
      tf.losses.add_loss(l1_loss * sparse_regularizer)

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

    # Fusion.
    video_seq_features = []
    for flag, feature in zip(
        [options.feature_config.use_rgb, 
         options.feature_config.use_flow, 
         options.feature_config.use_audio, 
         options.feature_config.use_gyroscope, 
         options.feature_config.use_accelerator], 
        [rgb_features, 
         flow_features, 
         audio_features, 
         gyroscope_features, 
         accelerator_features]):
      if flag:
        video_seq_features.append(feature)
      
    video_seq_features = tf.concat(video_seq_features, -1)
    video_seq_features = self._compute_video_features(video_seq_features, 'SeqFeature')

    # Compute attention.
    verb_attention, _ = self._compute_class_aware_attention(
        video_seq_features, 
        video_lengths, 
        n_classes=1 + options.n_verb_classes,
        use_gumbel_softmax=options.use_gumbel_softmax,
        gumbel_softmax_temperature=options.gumbel_softmax_temperature,
        attention_dropout_rate=options.attention_dropout_rate,
        sparse_regularizer=options.sparse_regularizer,
        is_training=self.is_training,
        name='VerbAttention')
    noun_attention, _ = self._compute_class_aware_attention(
        video_seq_features, 
        video_lengths, 
        n_classes=1 + options.n_noun_classes,
        use_gumbel_softmax=options.use_gumbel_softmax,
        gumbel_softmax_temperature=options.gumbel_softmax_temperature,
        attention_dropout_rate=options.attention_dropout_rate,
        sparse_regularizer=options.sparse_regularizer,
        is_training=self.is_training,
        name='NounAttention')

    # Apply attention.
    #   verb_features shape = [batch, n_verb_classes, dims].
    #   noun_features shape = [batch, n_noun_classes, dims].
    verb_features = tf.einsum('bcl,bld->bcd', verb_attention, video_seq_features)
    noun_features = tf.einsum('bcl,bld->bcd', noun_attention, video_seq_features)
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

    return {
        'video_id': video_ids,
        'video_lengths': video_lengths,
        'video_features': video_features,
        'video_seq_features': video_seq_features,
        'verb_features': verb_features,
        'verb_seq_attention': verb_attention,
        'verb_seq_classification': verb_seq_classification,
        'noun_features': noun_features,
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
    noun_features, verb_features = predictions['noun_features'], predictions['verb_features']
    noun_fc, verb_fc = self._layers[-2:]

    # Gather dims.
    batch = noun_ids.shape[0].value
    batch_id = tf.range(batch, dtype=tf.int32)
    noun_features = tf.gather_nd(noun_features, 
                                 tf.stack([batch_id, noun_ids], -1))
    verb_features = tf.gather_nd(verb_features, 
                                 tf.stack([batch_id, verb_ids], -1))

    # Classification.
    noun_logits = noun_fc(noun_features)
    verb_logits = verb_fc(verb_features)

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
    noun_seq_classification = predictions['noun_seq_classification']

    #######################################################
    # Detection evaluation.
    #######################################################

    video_id = predictions['video_id']
    video_lengths = predictions['video_lengths']
    video_features = predictions['video_features']
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
