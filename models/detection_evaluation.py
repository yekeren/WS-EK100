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
import pandas as pd
import tensorflow as tf

from evaluate_detection_json_ek100 import ANETdetection


def _load_id_to_name(file_name):
  df = pd.read_csv(file_name)
  return {i: v for i, v in zip(df['id'], df['key'])}

class DetectionEvaluator(object):
  """Class to evaluate scene graph generation metrics. """

  def __init__(self, eval_annotation_path, noun_classes_csv, verb_classes_csv):
    """Initializes the evaluator. 

    Args:
      eval_annotation_path: Path to the pkl file storing ground-truth.
      noun_classes_csv: Path to the noun classes csv file.
      verb_classes_csv: Path to the verb classes csv file.
    """
    self._video_detections = {}
    self._groundtruth_df = pd.read_pickle(eval_annotation_path)

    classid2noun = _load_id_to_name(noun_classes_csv)
    classid2verb = _load_id_to_name(verb_classes_csv)

    noun_ids = self._groundtruth_df['noun_class'].unique()
    verb_ids = self._groundtruth_df['verb_class'].unique()

    self._classid2noun = {i: classid2noun[cls] for i, cls in enumerate(noun_ids)}
    self._classid2verb = {i: classid2verb[cls] for i, cls in enumerate(verb_ids)}


  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._video_detections.clear()

  def evaluate(self):
    """Evaluates the generated scene graphs, returns recall metrics.

    Returns:
      A dictionary holding
      - `SceneGraphs/Recall@50`: Recall with 50 detections.
      - `SceneGraphs/Recall@100`: Recall with 100 detections.
    """
    logging.info('Performing evaluation on %d images.',
                 len(self._video_detections))

    submission = {
        'version': "0.2",
        'challenge': 'action_detection',
        'sls_pt': -1,
        'sls_tl': -1,
        'sls_td': -1,
        'results': self._video_detections}


    with open('results.json', 'w') as f:
      json.dump(submission, f)

    def dump(results, maps, task):
      for i, map in enumerate(maps):
        j=i+1
        results["metrics/%s_map_at_%02d" % (task, j)] = map
      return results

    metrics = {'metrics/n_example': len(self._video_detections)}
    for task, idmap in zip(['verb', 'noun', 'action'], 
                           [self._classid2verb, self._classid2noun, None]):
      evaluator = ANETdetection(self._groundtruth_df, submission, label=task)
      maps, avg = evaluator.evaluate()
      maps_per_class = evaluator.ap.mean(axis=0)
      dump(metrics, maps, task)
      metrics["metrics/%s_map_avg" % task] = avg
      if idmap:
        for class_id in range(len(maps_per_class)):
          metrics['%s_per_class_map/%s' % (task, idmap[class_id])] = maps_per_class[class_id]

    return metrics

  def add_single_detected_video_info(self, video_id, video_info):
    """Adds detections for a single video.

    Args:
      video_id: A unique identifier for the video.
      video_info: A dictionary containing the detection fields:
        t_start - Starting frame.
        t_end - Ending frame.
        verb_id - Class id.
        verb_score - Class score.
    """
    video_id = video_id.decode('ascii')
    if video_id in self._video_detections:
      logging.warning('Ignoring detection for %s.', video_id)
      return

    detections = []
    for t_start, t_end, action_id, action_score, verb_id, noun_id in zip(
        video_info[0], video_info[1], video_info[2], video_info[3], video_info[4], video_info[5]):
      detections.append({
        'verb': int(verb_id),
        'noun': int(noun_id),
        'action': '%i,%i' % (verb_id, noun_id),
        'score': float(action_score),
        'segment': [float(t_start), float(t_end)],
        })

    self._video_detections[video_id] = detections
    # logging.info('Adding video %s.', video_id)

  def add_eval_dict(self, eval_dict):
    """Observes an evaluation result dict for a single example.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating scene graph
        generation performance.

    Returns:
      An update_op that can be used to update the eval metrics in
        tf.estimator.EstimatorSpce.
    """

    def update_op(video_id_batched, 
                  n_detection_batched, 
                  t_start_batched, 
                  t_end_batched, 
                  action_id_batched, 
                  action_score_batched,
                  verb_id_batched, 
                  noun_id_batched):
      for video_id, n_detection, t_start, t_end, action_id, action_score, verb_id, noun_id in zip(
          video_id_batched, n_detection_batched, t_start_batched, 
          t_end_batched, action_id_batched, action_score_batched, verb_id_batched, noun_id_batched):

        t_start = t_start[:n_detection]
        t_end = t_end[:n_detection]
        action_id = action_id[:n_detection]
        action_score = action_score[:n_detection]
        verb_id = verb_id[:n_detection]
        noun_id = noun_id[:n_detection]

        self.add_single_detected_video_info(video_id,
            (t_start, t_end, action_id, action_score, verb_id, noun_id))

    video_id = eval_dict['video_id']
    n_detection = eval_dict['n_detection']
    t_start = eval_dict['t_start']
    t_end = eval_dict['t_end']
    action_id = eval_dict['action_id']
    action_score = eval_dict['action_score']
    verb_id = eval_dict['verb_id']
    noun_id = eval_dict['noun_id']

    return tf.py_func(update_op, [
      video_id, n_detection, t_start, t_end, action_id, action_score, verb_id, noun_id], [])

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns a dictionary of eval metric ops.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating scene graph
        generation performance.

    Returns:
      A dictionary of metric names to tule of value_op and update_op that can
        be used as eval metric ops in tf.estimator.EstimatorSpec.
    """
    update_op = self.add_eval_dict(eval_dict)
    metric_names = [
        'metrics/n_example',
        'metrics/verb_map_at_01',
        'metrics/verb_map_at_02',
        'metrics/verb_map_at_03',
        'metrics/verb_map_at_04',
        'metrics/verb_map_at_05',
        'metrics/verb_map_avg',
        'metrics/noun_map_at_01',
        'metrics/noun_map_at_02',
        'metrics/noun_map_at_03',
        'metrics/noun_map_at_04',
        'metrics/noun_map_at_05',
        'metrics/noun_map_avg',
        'metrics/action_map_at_01',
        'metrics/action_map_at_02',
        'metrics/action_map_at_03',
        'metrics/action_map_at_04',
        'metrics/action_map_at_05',
        'metrics/action_map_avg',
    ]
    metric_names.extend(['verb_per_class_map/%s' % x for x in self._classid2verb.values()])
    metric_names.extend(['noun_per_class_map/%s' % x for x in self._classid2noun.values()])

    def first_value_func():
      self._metrics = self.evaluate()
      self.clear()
      return np.float32(self._metrics[metric_names[0]])

    def value_func_factory(metric_name):
      def value_func():
        return np.float32(self._metrics[metric_name])
      return value_func

    first_value_op = tf.py_func(first_value_func, [], tf.float32)
    eval_metric_ops = {metric_names[0]: (first_value_op, update_op)}

    with tf.control_dependencies([first_value_op]):
      for metric_name in metric_names[1:]:
        eval_metric_ops[metric_name] = (tf.py_func(
            value_func_factory(metric_name), [], np.float32), update_op)

    return eval_metric_ops
