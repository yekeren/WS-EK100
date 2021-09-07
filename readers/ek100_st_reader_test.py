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
import ek100_st_reader

tf.compat.v1.enable_eager_execution()


class EK100STReaderTest(tf.test.TestCase):

  def test_get_input_fn(self):
    batch_size = 32
    options_str = r"""
      ek100_st_reader {
        path_to_annotations: "epic-kitchens-100-annotations/EPIC_100_train.pkl"
        path_to_video_lengths: "epic-kitchens-100-annotations/EPIC_100_video_info.csv"
        video_feature_directory: "data/video_features_s1/"
        meta_data_directory: "epic-kitchens-100/"
        audio_feature_directory: "data/audio_feature_files/"
        audio_embedding_directory: "data/audio_embedding_files/"
        cache_directory: "cache/"
        batch_size: %i
        shuffle_buffer_size: 50
        prefetch_buffer_size: 2
        random_offset_range: 5
      }
    """ % batch_size
    options = text_format.Merge(options_str, reader_pb2.Reader())

    dataset = reader.get_input_fn(options, is_training=True)()
    for elem in dataset.take(1):
      inputs, outputs = elem
      (video_ids, video_features, audio_features, gyroscope_features, accelerator_features,
       observation_windows, ws_noun_labels, ws_verb_labels) = inputs
      noun_labels, verb_labels = outputs

      self.assertAllClose(ws_noun_labels, noun_labels)
      self.assertAllClose(ws_verb_labels, verb_labels)

      max_observation_window = video_features.shape[1]
      self.assertAllEqual(noun_labels.shape, verb_labels.shape)
      self.assertEqual(max_observation_window, observation_windows.numpy().max())

      self.assertAllEqual(video_ids.shape, [batch_size])
      self.assertAllEqual(video_features.shape, [batch_size, max_observation_window, 2048])
      self.assertAllEqual(audio_features.shape, [batch_size, max_observation_window, 128])
      self.assertAllEqual(gyroscope_features.shape, [batch_size, max_observation_window, 3])
      self.assertAllEqual(accelerator_features.shape, [batch_size, max_observation_window, 3])
      self.assertAllEqual(observation_windows.shape, [batch_size])
      self.assertAllEqual(noun_labels.shape, [batch_size])
      self.assertAllEqual(verb_labels.shape, [batch_size])

      self.assertDTypeEqual(video_ids, np.object)
      self.assertDTypeEqual(video_features, np.float32)
      self.assertDTypeEqual(audio_features, np.float32)
      self.assertDTypeEqual(gyroscope_features, np.float32)
      self.assertDTypeEqual(accelerator_features, np.float32)
      self.assertDTypeEqual(observation_windows, np.int32)
      self.assertDTypeEqual(noun_labels, np.int32)
      self.assertDTypeEqual(verb_labels, np.int32)


if __name__ == '__main__':
  tf.test.main()
