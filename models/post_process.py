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
import tensorflow as tf


def _py_proposal_scoring(i_start, i_end, scores):
  """Heuristically scores the interval denoted by (i_start, i_end).

  Args:
    i_start: Index of the start segment.
    i_end: Index of the end segment.
    scores: Scores for all segments, [observation_window].
  """
  observation_window = len(scores)
  left = scores[i_start] if i_start == 0 else scores[i_start] - scores[i_start - 1]
  right = scores[i_end] if i_end == observation_window - 1 else scores[i_end] - scores[i_end + 1]
  #values = scores[i_start: 1 + i_end]
  return (left + right) / 2


def _py_post_process(scores, score_min, score_max, score_step):
  """Post processing, generate continuous detections based on segment scores.

  proposal = (i_start, i_end, confidence).

  Args:
    scores: Scores for all segments, [observation_window].
    score_min: Minimum score to consider.
    score_max: Maximum score possible.
    score_step: Stride step to attempt.

  Returns:
    detections: A list of (i_start, i_end, confidence).
  """
  all_intervals = set()
  score_threshold = score_min
  while score_threshold <= score_max:
    mask = scores >= score_threshold
    intervals = []  # A list of [i_start, i_end].
    for i, m in enumerate(mask):
      if m == 1:
        if not intervals or intervals[-1][1] != i - 1:
          intervals.append([i, i])
        else:
          intervals[-1][1] = i
    all_intervals |= set(map(tuple, intervals))
    score_threshold += score_step

  detections = [(i_start, i_end, _py_proposal_scoring(i_start, i_end, scores)) for i_start, i_end in all_intervals]
  return detections 


def py_post_process(all_scores, max_n_detection=100, score_min=0.1, score_max=1.0, score_step=0.1):
  """Post processing, generate continuous detections based on segment scores.

  Note that the all_scores here ruled out background.

  Args:
    all_scores: Scores for all segments, [batch, observation_window, n_classes].
    max_n_detection: Maximum number of detection segments.
    score_min: Minimum score to consider.
    score_max: Maximum score possible.
    score_step: Stride step to attempt.

  Returns:
    i_start: Start frame index, [max_n_detection] int array.
    i_end: End frame index, [max_n_detection] int array.
    class_id: Class id, [max_n_detection] int array.
    confidence: Detection score, [max_n_detection] float array.
  """
  detections = []
  for class_id, scores in enumerate(all_scores.transpose([1, 0])):
    per_class_detections = _py_post_process(scores, score_min, score_max, score_step)
    for i_start, i_end, confidence in per_class_detections:
      detections.append((i_start, i_end, class_id, confidence))
  detections.sort(key=lambda x: -x[-1])
  detections = detections[:max_n_detection]

  # Pack the results.
  i_start, i_end, class_id, confidence = zip(*detections)
  i_start = np.array(i_start, dtype=np.int32)
  i_end = np.array(i_end, dtype=np.int32)
  class_id = np.array(class_id, dtype=np.int32)
  confidence = np.array(confidence, dtype=np.float32)
  return i_start, i_end, class_id, confidence


def post_process(all_scores, max_n_detection=100, score_min=0.1, score_max=1.0, score_step=0.1):
  """Post processing, generate continuous detections based on segment scores.

  Note that the all_scores here ruled out background.

  Args:
    all_scores: Scores for all segments, [batch, observation_window, n_classes].
    max_n_detection: Maximum number of detection segments.
    score_min: Minimum score to consider.
    score_max: Maximum score possible.
    score_step: Stride step to attempt.

  Returns:
    n_detection: Number of detected segments, [batch] int array.
    i_start: Start frame index, [batch, max_n_detection] int array.
    i_end: End frame index, [batch, max_n_detection] int array.
    class_id: Class id, [batch, max_n_detection] int array.
    confidence: Detection score, [batch, max_n_detection] float array.
  """
  output_types = [tf.int32, tf.int32, tf.int32, tf.int32, tf.float32]


  def _py_video_post_process(all_scores):
    """Per video python processing."""
    detections = []
    for class_id, scores in enumerate(all_scores.transpose([1, 0])):
      per_class_detections = _py_post_process(scores, score_min, score_max, score_step)
      for i_start, i_end, confidence in per_class_detections:
        detections.append((i_start, i_end, class_id, confidence))
    detections.sort(key=lambda x: -x[-1])

    n_detection = len(detections)
    if n_detection < max_n_detection:
      detections.extend([(-1, -1, -1, 0) for _ in range(max_n_detection - n_detection)])
    detections = detections[:max_n_detection]

    i_start, i_end, class_id, confidence = zip(*detections)
    i_start = np.array(i_start, dtype=np.int32)
    i_end = np.array(i_end, dtype=np.int32)
    class_id = np.array(class_id, dtype=np.int32)
    confidence = np.array(confidence, dtype=np.float32)
    return np.cast[np.int32](n_detection), i_start, i_end, class_id, confidence


  def _per_video_post_process(scores):
    """Per video TF processing."""
    return tf.py_func(_py_video_post_process, [scores], output_types)


  (n_detection, i_start, i_end, class_id, confidence) = tf.map_fn(
      _per_video_post_process, 
      elems=all_scores, 
      dtype=output_types, 
      parallel_iterations=32, 
      back_prop=False)

  batch = all_scores.shape[0]
  n_detection.set_shape([batch])
  i_start.set_shape([batch, max_n_detection])
  i_end.set_shape([batch, max_n_detection])
  class_id.set_shape([batch, max_n_detection])
  confidence.set_shape([batch, max_n_detection])

  return [n_detection, i_start, i_end, class_id, confidence]
