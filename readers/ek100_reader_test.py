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

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from protos import reader_pb2

import reader
import ek100_reader

tf.compat.v1.enable_eager_execution()


class EK100ReaderTest(tf.test.TestCase):

  def test_get_input_fn(self):
    batch_size = 32
    options_str = r"""
      ek100_reader {
        path_to_annotations: "epic-kitchens-100-annotations/EPIC_100_validation.pkl"
        path_to_video_lengths: "epic-kitchens-100-annotations/EPIC_100_video_info.csv"
        feature_directory: "data/video_features_s1/"
        batch_size: %i
        shuffle_buffer_size: 50
        prefetch_buffer_size: 2
      }
    """ % batch_size
    options = text_format.Merge(options_str, reader_pb2.Reader())

    dataset = reader.get_input_fn(options, is_training=False)()
    for elem in dataset.take(1):
      inputs, outputs = elem
      video_ids, video_features, observation_windows = inputs
      noun_labels, verb_labels = outputs
      max_observation_window = video_features.shape[1]
      feature_dims = video_features.shape[2]

      self.assertAllEqual(video_ids.shape, [batch_size])
      self.assertAllEqual(video_features.shape, [batch_size, max_observation_window, feature_dims])
      self.assertAllEqual(observation_windows.shape, [batch_size])
      self.assertAllEqual(noun_labels.shape, [batch_size, max_observation_window])
      self.assertAllEqual(verb_labels.shape, [batch_size, max_observation_window])
      self.assertEqual(max_observation_window, observation_windows.numpy().max())

      self.assertDTypeEqual(video_ids, np.object)
      self.assertDTypeEqual(video_features, np.float32)
      self.assertDTypeEqual(observation_windows, np.int32)
      self.assertDTypeEqual(noun_labels, np.int32)
      self.assertDTypeEqual(verb_labels, np.int32)


if __name__ == '__main__':
  tf.test.main()
