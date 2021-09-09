# WS-EK100 - A weakly supervised video action detection model

* [0 Overview](#0-overview)
* [1 Installation](#1-installation)
  - [1.1 Audio Features](#11-audio-features)
  - [1.2 Video Features](#12-video-features)
* [2 Model Training](#2-model-training)


## 0 Overview

## 1 Installation

```
git clone "https://github.com/yekeren/WS-EK100.git"
cd "WS-EK100" && mkdir "data"
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

Our offline video feature extraction process is primarily based on [https://github.com/epic-kitchens/C2-Action-Detection](https://github.com/epic-kitchens/C2-Action-Detection). So, it is recommended to get yourself familiar with the C2-Action-Detection repository. Our document is self-contained and we shall reiterate steps in [C2-Action-Detection](https://github.com/epic-kitchens/C2-Action-Detection), so, readers only need to refer to our document for extracting video features (when possible, refer to [C2-Action-Detection](https://github.com/epic-kitchens/C2-Action-Detection) for the missing details).

First, clone and checkout **our modified** [C2-Action-Detection](https://github.com/epic-kitchens/C2-Action-Detection):
```
git clone https://github.com/yekeren/C2-Action-Detection
git checkout yekeren/ek100-exporter
```

Next, download the full set of features:
```
cd C2-Action-Detection/BMNProposalGenerator
sh scripts/download_data_ek100_full.sh
```

The above process shall generate two .mdb files ```data/ek100/rgb/data.mdb``` and ```data/ek100/flow/data.mdb```, storing the full set of features.

Then, we create the PyTorch enviroment for feature extraction:
```
conda env create -f environment.yml
conda activate c2-action-detection-bmn
```


Then, we sample **1 frame per second** using the full set. We assume ```data/ek100/video_features/``` to be the output feature directory (.npy video feature files will be generated here).

```
python export_video_features.py \
  --path_to_dataset data/ek100 \
  --path_to_video_features data/ek100/video_features/ \
  --rgb_lmdb data/ek100/rgb \
  --flow_lmdb data/ek100/flow
```

Finally, we go back to the root directory (```WS-EK100```) from our current path (```WS-EK100/C2-Action-Detection/BMNProposalGenerator```), and change back to our original python environment (e.g., ```conda activate ek100```).

```
cd ../../
conda activate ek100  # An example, change to the actual enviroment.
```

## 2 Model Training

