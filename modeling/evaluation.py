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

from absl import app
from absl import flags
from absl import logging

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf


from google.protobuf import text_format
from models.post_process import py_post_process as post_process
from modeling import trainer
from protos import model_pb2
from protos import pipeline_pb2
from readers import reader
from scipy.special import softmax

from evaluate_detection_json_ek100 import ANETdetection

flags.DEFINE_string('model_dir', 'logs/fs_det/',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('video_lengths_csv', 'epic-kitchens-100-annotations/EPIC_100_video_info.csv',
                    'Path to the csv file storing video lengths.')

flags.DEFINE_string('verb_classes_csv', 'epic-kitchens-100-annotations/EPIC_100_verb_classes.csv',
                    'Path to the csv file storing verb classes.')

flags.DEFINE_string('noun_classes_csv', 'epic-kitchens-100-annotations/EPIC_100_noun_classes.csv',
                    'Path to the csv file storing noun classes.')

flags.DEFINE_string('groundtruth_df_path', 'epic-kitchens-100-annotations/EPIC_100_validation.pkl',
                    'Path to the pkl file storing ground-truth.')


FLAGS = flags.FLAGS



def load_id_to_name(file_name):
  """Loads id to name csv file."""
  df = pd.read_csv(file_name)
  return {i + 1: v for i, v in zip(df['id'], df['key'])}


def load_video_lengths(file_name):
  df = pd.read_csv(file_name, sep=',')
  return {i: l for i, l in zip(df['video_id'], df['duration'])}


def main(_):
  logging.set_verbosity(logging.DEBUG)

  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    
    with tf.io.gfile.GFile(os.path.join(FLAGS.model_dir, 'pipeline.pbtxt'), 'r') as fp:
      pipeline_proto = text_format.Merge(fp.read(), pipeline_pb2.Pipeline())
      
    # Build the TF model and input pipe.
    input_fn = reader.get_input_fn(pipeline_proto.eval_reader, is_training=False)
    model_fn = trainer.create_model_fn(pipeline_proto)
      
    features, labels = input_fn().make_one_shot_iterator().get_next()
    predictions = model_fn(features, labels, tf.estimator.ModeKeys.PREDICT, None).predictions
      
    def data_generator():
      saver = tf.train.Saver()
      with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        while True:
          yield sess.run([predictions, labels])
                            
    dg = data_generator()

    # Load noun/verb classes.
    classid2verb = load_id_to_name(FLAGS.verb_classes_csv)
    classid2noun = load_id_to_name(FLAGS.noun_classes_csv)

    # Load video_lengths
    video_lengths = load_video_lengths(FLAGS.video_lengths_csv)

    # Post processing.
    results = {}

    count = 0
    while True:
      try:
        y_pred, _ = next(dg)
        video_id = y_pred['video_id'][0].decode('ascii')
        noun_scores = softmax(y_pred['noun_logits'][0], -1)
        verb_scores = softmax(y_pred['verb_logits'][0], -1)

        (i_start_array, i_end_array, class_id_array,
         confidence_array) = post_process(verb_scores[:, 1:], 
                                          max_n_detection=100, 
                                          score_min=0.05, 
                                          score_max=0.25, 
                                          score_step=0.05)

        observation_window = verb_scores.shape[0]
        duration_per_segment = video_lengths[video_id] / observation_window

        results[video_id] = []
        logging.info(video_id)

        for i_start, i_end, verb_id, score in zip(
            i_start_array, i_end_array, class_id_array, confidence_array):

          results[video_id].append({
            'verb': int(verb_id),
            'noun': 0,
            'action': '0,0',
            'score': float(score),
            'segment': [i_start * duration_per_segment, (i_end + 1) * duration_per_segment],
            })
        
        count += 1
      except tf.errors.OutOfRangeError:
        break

    logging.info('Finished packing submission')

    # Create submission and evaluate.
    submission = {
        'version': "0.2",
        'challenge': 'action_detection',
        'sls_pt': -1,
        'sls_tl': -1,
        'sls_td': -1,
        'results': results}

    with open('results.json', 'w') as f:
      json.dump(submission, f)

    def dump(results, maps, task):
      for i, map in enumerate(maps):
        j=i+1
        results[f"{task}_map_at_{j:02d}"] = map*100
      return results

    def print_metrics(metrics):
      for name, value in metrics.items():
        print("{name}: {value:0.2f}".format(name=name, value=value))

    groundtruth_df = pd.read_pickle(FLAGS.groundtruth_df_path)

    display_metrics = {}
    for task in ['verb', 'noun', 'action']:
      maps, avg = ANETdetection(groundtruth_df, submission, label=task).evaluate()
      dump(display_metrics, maps, task)
      display_metrics[f"{task}_map_avg"] = avg * 100

    for sls in ["sls_pt", "sls_tl", "sls_td"]:
      display_metrics[sls] = submission[sls]

    print_metrics(display_metrics)
    logging.info('Finished evaluating %i examples.', count)


if __name__ == '__main__':
  app.run(main)
