Currently, Image-Bind could not be started, there is a problem with the cuda drivers. The inference was conducted only for the LLama Adapter and other models.


## Equipment and environment

**Python** 3.10.12

**Cuda compilation tools** release 12.8, V12.8.61

**Nvidia A100 80gb**

The environment differs from the author's, so requirements are not stably installed, but the specified file is assembled on such equipment.


## Setup

* Setup up a new conda env. Install ImageBind and other necessary packages.
  ```bash
  # create venv
  python3 -m venv .venv
  source .venv/bin/activate

  pip install -r requirements.txt

  cd measure
  python3 [model]_[dataset].py
  ```


## Dataset brief overview

### CUTE80
https://github.com/Jyouhou/Case-Sensitive-Scene-Text-Recognition-Datasets

CUTE80 is a dataset for text recognition in natural scene images, containing 80(**288**) high-resolution images with complex backgrounds and diverse text styles. It includes annotations for word-level bounding boxes and transcriptions, suitable for evaluating text detection and recognition models.

### ICDAR2013
https://rrc.cvc.uab.es/?ch=2&com=downloads 2.3 test set

The ICDAR2013 dataset, from the International Conference on Document Analysis and Recognition, focuses on scene text detection and recognition. It contains 229 training and 233 testing**1095 in fact** images with word-level annotations, featuring varied fonts, sizes, and orientations in real-world settings.

### ICDAR2015
https://rrc.cvc.uab.es/?ch=2&com=downloads - 2.4 пришлось взять train, test не имел разметки

ICDAR2015, part of the Robust Reading Competition, includes 1,000 training and 500 testing**229infact** images captured with wearable devices. It provides word-level bounding box annotations and transcriptions, emphasizing incidental scene text with challenges like motion blur and low resolution.

### IIIT5K
https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset

The IIIT5K-Word dataset contains 5,000 cropped word images from natural scenes, split into 2,000 training and 3,000 testing images**их и взяли**. Each image includes word-level transcriptions and is designed for text recognition tasks, with diverse fonts, colors, and backgrounds.

### SVT
https://github.com/Jyouhou/Case-Sensitive-Scene-Text-Recognition-Datasets

The Street View Text (SVT) dataset consists of 350 images collected from Google Street View, with 647 word-level bounding boxes and transcriptions[images]. It is designed for text detection and recognition in outdoor environments, featuring challenges like illumination variations and occlusions.

### SVTP
https://github.com/Jyouhou/Case-Sensitive-Scene-Text-Recognition-Datasets

SVTP (Street View Text Perspective) is an extension of SVT, containing 645 cropped word images with perspective distortions. It includes transcriptions and is tailored for evaluating text recognition under challenging geometric transformations.

### VQA

The Visual Question Answering (VQA) dataset includes images paired with natural language questions and answers, focusing on text in visual contexts. It contains over 200,000**3289** images and 1.1 million questions, suitable for tasks combining text recognition and reasoning.

### WordArt
The WordArt dataset comprises 1,500(**1511**) synthetic images with artistic text styles, including varied fonts, colors, and effects. It provides word-level annotations and is designed for training and evaluating text recognition models on stylized text.


## Measurements

### General provisions

All measurements are located in the datasets folder.
..._metrics.json - calculation of inference statistics
..._results.json - markup obtained during the inference


### old_measurations
the results that were not included in the folders with datasets were important at the beginning for choosing the metrics used by the authors: case_insensitive_accuracy, word_accuracy, strict_word_accuracy, small_lexicon_accuracy, ...
Were the reason for choosing the implementation of word accuracy specified in the presentation.
