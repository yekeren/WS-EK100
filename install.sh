#!/bin/sh

set -o errexit
set -o nounset
set -x

if [ ! -d "tensorflow_models" ]; then
  git clone "https://github.com/tensorflow/models.git" "tensorflow_models" 
fi

mkdir -p "data"
if [ ! -f "data/vggish_model.ckpt" ]; then
  wget -O "data/vggish_model.ckpt" \
    "https://storage.googleapis.com/audioset/vggish_model.ckpt"
fi
if [ ! -f "data/vggish_pca_params.npz" ]; then
  wget -O "data/vggish_pca_params.npz" \
    "https://storage.googleapis.com/audioset/vggish_pca_params.npz"
fi
