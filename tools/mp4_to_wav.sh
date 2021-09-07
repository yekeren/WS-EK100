#!/bin/sh

set -o errexit
set -o nounset
set -x

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 INPUT_DIR OUTPUT_DIR"
  exit 1
fi

input_dir=$1
output_dir=$2

mkdir -p ${output_dir}
count=0
find "${input_dir}/" -type f -name "*.MP4" | while read filepath; do
  count=$((count+1))
  mp4file=$(basename "${filepath}")
  wavfile=${mp4file//MP4/wav} 
  </dev/null ffmpeg -i "${filepath}" "${output_dir}/${wavfile}" > "log/ffmpeg_$((count%20)).log" 2>&1 &
  if [ $((count%20)) -eq 0 ]; then
    wait
  fi
done
