# WS-EP100 - A weakly supervised video action detection model

* [0 Overview](#0-overview)
* [1 Installation](#1-installation)
  - [1.1 Audio Features](#11-audio-features)
  - [1.2 Video Features](#12-video-features)


## 0 Overview

## 1 Installation

```
git clone "https://github.com/yekeren/WS-EP100.git"
cd "WS-EP100" && mkdir "data"
```

We use Tensorflow 1.5 and Python 3.6.4. To continue, please ensure that at least the correct Python version is installed.
[requirements.txt](requirements.txt) defines the list of python packages we installed.
Simply run ```pip install -r requirements.txt``` to install these packages after setting up python.
Next, run ```protoc protos/*.proto --python_out=.``` to compile the required protobuf protocol files, which are used for storing configurations.

```
pip install -r requirements.txt
protoc protos/*.proto --python_out=.
```

### 1.1 Audio Features

We use the [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) pre-trained on the [AudioSet dataset](https://research.google.com/audioset/) to extract the audio semantic features (128-D).
First, checkout the implementation of the VGGish model:

```
git clone "https://github.com/tensorflow/models.git" "tensorflow_models" 
```

Then, download the pre-trained model checkpoint.

```
wget -O "data/vggish_model.ckpt" "https://storage.googleapis.com/audioset/vggish_model.ckpt"
```

Next, extract the .wav files from the .mp4 videos, using the following command (assuming ```data/audio_wav_files/``` to be the directory storing .wav files).
The PATH_TO_THE_VIDEO_DIR should point to the root directory of the Epic Kitchens Dataset (the directory contain subfolders such as ```P01```, ```P02```, etc. and .mp4 files are within, e.g., ```/afs/cs.pitt.edu/projects/kovashka/epic-kitchens/epic-kitchens-100/```).

```
sh tools/mp4_to_wav.sh PATH_TO_THE_VIDEO_DIR "data/audio_wav_files/"
```

Finally, extract the 128-D audio semantic embeddings using the following command (assuming ```data/audio_feature_files/``` and ```data/audio_embedding_files/``` to be the directories storing the original audio features and semantic embeddings, respectively)

```
python "tools/wav_to_emb.py" \
  --audio_dir "data/audio_wav_files/" \
  --output_feature_dir "data/audio_feature_files/" \
  --output_embedding_dir "data/audio_embedding_files/" \
  --checkpoint_path "data/vggish_model.ckpt"
```
### 1.2 Video Features
