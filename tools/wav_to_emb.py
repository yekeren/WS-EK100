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

from absl import app
from absl import flags
from absl import logging

import os
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_slim


flags.DEFINE_string('audio_dir', 'data/audio_wav_files',
                    'Path to the directory storing wav files.')

flags.DEFINE_string('checkpoint_path', 'data/vggish_model.ckpt',
                    'Path to the pre-trained vggish checkpoint.')

flags.DEFINE_string('output_feature_dir', 'data/audio_feature_files',
                    'Path to the directory storing audio features.')

flags.DEFINE_string('output_embedding_dir', 'data/audio_embedding_files',
                    'Path to the directory storing audio embeddings.')


FLAGS = flags.FLAGS

def main(_):
  logging.set_verbosity(logging.DEBUG)

  tf.gfile.MakeDirs(FLAGS.output_feature_dir)
  tf.gfile.MakeDirs(FLAGS.output_embedding_dir)

  # Initialize tensorflow graph.
  with tf.Graph().as_default(), tf.Session() as sess:
    input_tensor = tf.placeholder(tf.float32, 
                                  shape=(None, 
                                         vggish_params.NUM_FRAMES,
                                         vggish_params.NUM_BANDS),
                                  name='input_features')
    output_tensor = vggish_slim.define_vggish_slim(input_tensor, 
                                                   training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint_path)

    # Define the embedding function.
    def _compute_audio_embedding(input_feature):
      """Computes audio embedding. """
      output_feature = sess.run(output_tensor, 
                                feed_dict={input_tensor: input_feature})
      return output_feature

    for file_name in os.listdir(FLAGS.audio_dir):
      if file_name.endswith('.wav'):
        out_feature_path = os.path.join(FLAGS.output_feature_dir,
                                        file_name.replace('.wav', '.npy'))
        out_emb_path = os.path.join(FLAGS.output_embedding_dir,
                                    file_name.replace('.wav', '.npy'))
        if os.path.isfile(out_feature_path) and os.path.isfile(out_emb_path):
          logging.info('Skip %s and %s.', out_feature_path, out_emb_path)
          continue

        # Read the raw audio features.
        #   raw_feature = [audio_length, 96, 64].
        raw_feature = vggish_input.wavfile_to_examples(
            os.path.join(FLAGS.audio_dir, file_name))
        np.testing.assert_equal(
            raw_feature.shape[1:],
            [vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

        # Compute audio embeddings.
        embedding = _compute_audio_embedding(raw_feature)

        np.save(out_feature_path, raw_feature)
        np.save(out_emb_path, embedding)
        logging.info('Finished processing %s and %s.', out_feature_path, out_emb_path)

  logging.info('Done')

if __name__ == '__main__':
  app.run(main)
