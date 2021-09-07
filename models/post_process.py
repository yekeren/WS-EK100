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

import json
import numpy as np
import tensorflow as tf
from scipy.ndimage import uniform_filter1d

with open('action_id.txt', 'r') as f:
  all_action_ids = json.load(f)


def _py_proposal_scoring(i_start, i_end, scores):
  """Heuristically scores the interval denoted by (i_start, i_end).

  Args:
    i_start: Index of the start segment.
    i_end: Index of the end segment.
    scores: Scores for all segments, [observation_window].

  Returns:
    Score for the segment [i_start, i_end].
  """
  # observation_window = len(scores)
  # left = scores[i_start] if i_start == 0 else scores[i_start] - scores[i_start - 1]
  # right = scores[i_end] if i_end == observation_window - 1 else scores[i_end] - scores[i_end + 1]
  # return (left + right) / 2
  values = scores[i_start: 1 + i_end]
  return values.mean()


def _non_maximum_suppression(detections, iou_threshold=0.4):
  """Non-maximum suppression.

  Args:
    detections: A list of (i_start, i_end, confidence).

  Returns:
    detections: NMS results, a list of (i_start, i_end, confidence).
  """
  def _length(segment):
    return max(0.0, segment[1] - segment[0])

  def _intersect(segment1, segment2):
    min1, max1 = segment1[0], segment1[1]
    min2, max2 = segment2[0], segment2[1]
    minv = max(min1, min2)
    maxv = min(max1, max2)
    return (minv, maxv)

  def _union(segment1, segment2):
    return _length(segment1) + _length(segment2) - _intersect(segment1, segment2)

  def _iou(segment1, segment2):
    intersect = _length(_intersect(segment1, segment2))
    union = _length(segment1) + _length(segment2) - intersect
    return intersect / (1e-12 + union)

  detections.sort(key=lambda x: -x[-1])

  results = []
  for det in detections:
    i = 0
    while i < len(results):
      iou = _iou(det, results[i])
      if iou >= iou_threshold:
        break
      i += 1
    if i == len(results):
      results.append(det)

  return results


def _py_post_process(action_scores, thresholds):
  """Post processing, generate continuous detections based on segment scores.

  proposal = (i_start, i_end, confidence).

  Args:
    scores: Scores for all segments, [observation_window].
    thresholds: A list of threshold values.

  Returns:
    detections: A list of (i_start, i_end, action_score).
  """
  action_scores = uniform_filter1d(action_scores, 3)

  all_intervals = set()

  for score_threshold in thresholds:
    mask = action_scores >= score_threshold
    intervals = []  # A list of [i_start, i_end].
    for i, m in enumerate(mask):
      if m == 1:
        if not intervals or intervals[-1][1] != i - 1:
          intervals.append([i, i])
        else:
          intervals[-1][1] = i
    all_intervals |= set(map(tuple, intervals))

  detections = [(i_start, i_end, _py_proposal_scoring(i_start, i_end, action_scores)) for i_start, i_end in all_intervals]
  detections = _non_maximum_suppression(detections)
  return detections 


def py_post_process(action_scores, max_n_detection=100, thresholds=[0.1, 0.2, 0.3, 0.4]):
  """Post processing, generate continuous detections based on segment scores.

  Note that the all_scores here ruled out background.

  Args:
    action_scores: Scores for all segments, [batch, observation_window, n_classes].
    max_n_detection: Maximum number of detection segments.
    thresholds: A list of threshold values.

  Returns:
    i_start: Start frame index, [max_n_detection] int array.
    i_end: End frame index, [max_n_detection] int array.
    class_id: Class id, [max_n_detection] int array.
    confidence: Detection score, [max_n_detection] float array.
  """
  detections = []

  for action_id in all_action_ids:
    per_class_detections = _py_post_process(action_scores[:, action_id], thresholds)
    for i_start, i_end, confidence in per_class_detections:
      detections.append((i_start, i_end, action_id, confidence))

  detections.sort(key=lambda x: -x[-1])
  detections = detections[:max_n_detection]

  # Pack the results.
  i_start, i_end, action_id, confidence = zip(*detections)
  i_start = np.array(i_start, dtype=np.int32)
  i_end = np.array(i_end, dtype=np.int32)
  action_id = np.array(action_id, dtype=np.int32)
  confidence = np.array(confidence, dtype=np.float32)
  return i_start, i_end, action_id, confidence


def post_process(action_scores, max_n_detection=100, thresholds=[0.1, 0.2, 0.3, 0.4]):
  """Post processing, generate continuous detections based on segment scores.

  Note that the all scores here ruled out background.

  Args:
    action_scores: Scores for all segments, [batch, observation_window, n_verb_classes * n_noun_classes].
    max_n_detection: Maximum number of detection segments.
    thresholds: A list of threshold values.

  Returns:
    n_detection: Number of detected segments, [batch] int array.
    i_start: Start frame index, [batch, max_n_detection] int array.
    i_end: End frame index, [batch, max_n_detection] int array.
    action_id: Class id, [batch, max_n_detection] int array.
    action_score: Detection score, [batch, max_n_detection] float array.
  """
  output_types = [tf.int32, tf.int32, tf.int32, tf.int32, tf.float32]


  def _py_video_post_process(action_scores):
    """Per video python processing."""
    detections = []

    for action_id in all_action_ids:
      per_class_detections = _py_post_process(action_scores[:, action_id], thresholds)
      for i_start, i_end, action_score in per_class_detections:
        detections.append((i_start, i_end, action_id, action_score))


    detections.sort(key=lambda x: -x[3])

    n_detection = len(detections)
    if n_detection < max_n_detection:
      detections.extend([(-1, -1, -1, 0) for _ in range(max_n_detection - n_detection)])
    detections = detections[:max_n_detection]
    n_detection = min(n_detection, max_n_detection)

    i_start, i_end, action_id, action_score = zip(*detections)
    i_start = np.array(i_start, dtype=np.int32)
    i_end = np.array(i_end, dtype=np.int32)
    action_id = np.array(action_id, dtype=np.int32)
    action_score = np.array(action_score, dtype=np.float32)
    return np.cast[np.int32](n_detection), i_start, i_end, action_id, action_score


  def _per_video_post_process(elems):
    """Per video TF processing."""
    return tf.py_func(_py_video_post_process, elems, output_types)


  (n_detection, i_start, i_end, action_id, action_score) = tf.map_fn(
      _per_video_post_process, 
      elems=[action_scores],
      dtype=output_types, 
      parallel_iterations=100, 
      back_prop=False)

  batch = action_scores.shape[0]
  n_detection.set_shape([batch])
  i_start.set_shape([batch, max_n_detection])
  i_end.set_shape([batch, max_n_detection])
  action_id.set_shape([batch, max_n_detection])
  action_score.set_shape([batch, max_n_detection])

  return [n_detection, i_start, i_end, action_id, action_score]
