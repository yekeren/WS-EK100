#!/bin/sh

set -o errexit
set -o nounset
set -x

export PYTHONPATH="`pwd`"
export PYTHONPATH="`pwd`/vggish:$PYTHONPATH"
export PYTHONPATH="`pwd`/C2-Action-Detection/EvaluationCode:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 MODEL_NAME GPU"
  exit 1
fi

model_name=$1
gpu=$2

export CUDA_VISIBLE_DEVICES="${gpu}"
python "modeling/trainer_main.py" \
  --alsologtostderr \
  --pipeline_proto="configs/${model_name}.pbtxt" \
  --model_dir="logs/${model_name}" \
  >> "log/${model_name}.log" 2>&1 &
