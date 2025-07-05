# 🖼️ Image Captioning Model
A model for generating textual description/ Speech of a given image based on the objects and actions in the image  Topics

This project implements an Image Captioning system using deep learning techniques. The model generates descriptive captions for images by combining image features and text sequences.

This project uses the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

Due to size constraints, the dataset is **not included** in this repository

Aproached Techniques:
Pre-extracted image features from a CNN (VGG16).

Sequence modeling using LSTM for caption generation.

Encoder-Decoder architecture combining image and text features.

## 🏗️ Model Architecture
Encoder:

Dense layers applied to pre-extracted image features (4096-dimensional vectors).

Decoder:

Embedding layer for text input.

LSTM for sequence modeling.

Dense layers for output prediction.

Loss Function:
(Categorical Crossentropy)

Optimizer:
(Adam)

## 📝 Dataset
Flickr8K dataset with corresponding captions.

Image features are pre-extracted and loaded during training.

## 🏷️ Key Features
Image-Text fusion with Encoder-Decoder LSTM.

Batch-wise data generator for memory efficiency.

Supports training with custom datasets.

## 📈 Example Output
Input Image	Generated Caption
"A man riding a horse on a beach."

License
This project is open-source and available under the MIT License.
