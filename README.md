# Lip Reading Deep Learning Project (LipNet Inspired)

This repository implements a deep learning-based lip-reading system that converts silent video footage of a speaker’s mouth into text. The model is inspired by the original LipNet architecture and processes raw video frames using spatiotemporal convolutions followed by recurrent sequence modeling. A Streamlit-based web interface is included to allow interactive testing through video uploads.

---

## 1. Introduction

Lip reading aims to recognize speech using only visual information by analyzing lip movements. Beyond speech recognition, the method is valuable in applications such as:

- Communication in noisy environments
- Assistive technology for speech-impaired users
- Silent speech interfaces
- Surveillance and security contexts

This implementation uses:

- Grayscale cropped video frames of the mouth region
- 3D CNN layers for spatial-temporal feature extraction
- Bidirectional LSTMs for sequential modeling
- CTC decoding to map frame-wise predictions to text

The training pipeline was based on GRID dataset-style `.mpg` videos.

---

## 2. Project Structure

|-- app_upload.py # Streamlit app for uploading and predicting from videos
|-- utils.py # Video loading, preprocessing, vocabulary utilities
|-- modelutil.py # Model architecture and checkpoint loading
|-- models-checkpoint/ # Model weights in TensorFlow checkpoint format
|-- requirements.txt # Dependencies for running the project


This repository assumes that model weights are already trained and stored locally.

---

## 3. Approach & Model Architecture

### 3.1 Preprocessing

Each video is processed frame-by-frame using OpenCV and TensorFlow:

- Frames converted to grayscale
- Mouth region cropped: `(190:236, 80:220)`
- Normalization performed across frames
- Output fed to model with shape: `(T, 46, 140, 1)`

This ensures consistent size and lighting across samples.

### 3.2 Vocabulary Encoding

Characters are converted to numeric tokens using `StringLookup`:


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = StringLookup(vocabulary=vocab)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), invert=True)

The decoder uses CTC to map time-distributed probabilities to text sequences.

### 3.3 Model Summary

The architecture contains:

Multiple Conv3D layers with temporal-spatial feature learning

Pooling to reduce spatial resolution across frames

TimeDistributed(Flatten()) to convert per-frame tensors into sequences

Bidirectional LSTMs to learn sequential motion patterns

Dense output layer with softmax over vocabulary tokens

Model weights are loaded externally via TensorFlow checkpoint files.

## 4. Streamlit Application

The included Streamlit app enables interaction through browser-based video uploads.

Key features:

Upload .mp4, .avi, .mkv, .mpg files

View uploaded video

Display preprocessed grayscale cropped frames

Display predicted character tokens

Output final decoded text

## 5. Model Weights

The model weights are stored as a TensorFlow checkpoint:

models-checkpoint/
    checkpoint
    checkpoint.data-00000-of-00001
    checkpoint.index

In modelutil.py:

weights_path = os.path.join("models-checkpoint", "checkpoint")
model.load_weights(weights_path)

## 6. Dataset Notes

The model was originally trained on the GRID corpus format, which contains:

Fixed-length .mpg videos

Consistent speaker format

Alignment files per utterance

## 7. Limitations

Model performance depends heavily on video framing and clarity

Works best on short, fixed-length utterances similar to training format

Not speaker-independent without additional training

No audio is used; purely visual inference

## 8. Future Enhancements

Convert checkpoint into a .keras or .h5 unified weight format

Add in-browser camera recording

Support variable-length clips via dynamic padding

Extend training to more diverse datasets
