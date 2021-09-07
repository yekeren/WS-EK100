#!/bin/bash

set -o errexit
set -o nounset
set -x

export PYTHONPATH="`pwd`"
export PYTHONPATH="`pwd`/vggish:$PYTHONPATH"
export PYTHONPATH="`pwd`/C2-Action-Detection/EvaluationCode:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"

models=(
  fully
  fully_noaudio
  fully_noflow
  fully_norgb
  fully_onlyaudio
  fully_onlyflow
  fully_onlyrgb
  single
  single_noaudio
  single_noflow
  single_norgb
  single_onlyrgb
)

for model in ${models[@]}; do
  python modeling/trainer_main.py --job test \
    --pipeline_proto "logs/${model}/pipeline.pbtxt" \
    --model_dir "logs/${model}/"
done
