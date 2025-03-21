# Transfer Learning Experiments with TensorFlow
[02_Transfer_Learning_Fine_Tuning.ipynb](https://github.com/Prashanna-Raj-Pandit/Transfer-Learning--TensorFlow/blob/main/02_Transfer_Learning_Fine_Tuning.ipynb)


This project conducts a series of transfer learning experiments using TensorFlow on the Food101 dataset, specifically the 10% subset of 10 food classes. The goal is to compare feature extraction and fine-tuning approaches with varying data percentages and data augmentation, leveraging the Functional API, augmentation techniques, and model checkpointing.

## Experiments Overview

The notebook implements the following experiments:
1. **Model 0**: Feature extraction with 10% of training data, no data augmentation.
2. **Model 1**: Feature extraction with 1% of training data, with data augmentation.
3. **Model 2**: Feature extraction with 10% of training data, with data augmentation.
4. **Model 3**: Fine-tuning with 10% of training data, with data augmentation.
5. **Model 4**: Fine-tuning with 100% of training data, with data augmentation.

## Prerequisites

- Python 3.x
- TensorFlow 2.18.0
- Access to a GPU (optional but recommended for faster training)
- Internet connection (for downloading dataset and helper functions)

## Setup Instructions

1. **Clone the Repository** (if applicable):
   Clone the repository and navigate to the project directory.

2. **Install Dependencies**:
   Install TensorFlow 2.18.0 using pip.

3. **Download Helper Functions**:
   Download custom helper functions from a specified GitHub repository using wget.

4. **Download Dataset**:
   Fetch the 10% subset of the Food101 dataset (10 classes) from a Google Storage URL and extract it using the provided `unzip_data` function.

## Dataset Structure

- **Training Data**: 750 images (75 per class, 10 classes)
- **Test Data**: 2500 images (250 per class, 10 classes)
- **Classes**: `chicken_curry`, `chicken_wings`, `fried_rice`, `grilled_salmon`, `hamburger`, `ice_cream`, `pizza`, `ramen`, `steak`, `sushi`
- **Image Size**: 224x224 pixels

## Model Creation

Models are constructed using TensorFlow's Functional API. The process starts by selecting a pre-trained base model from `tf.keras.applications`, such as EfficientNetV2B0, with the top classification layer excluded. The base model's weights are frozen for feature extraction to preserve pre-learned patterns. An input layer is defined with a shape of 224x224x3, followed by passing the inputs through the base model. The output is then aggregated using a GlobalAveragePooling2D layer to reduce dimensionality and computational load. Finally, a dense output layer with 10 units (one per class) and softmax activation is added. The model is compiled with categorical cross-entropy loss, the Adam optimizer, and accuracy as the metric.

## Data Augmentation

For experiments requiring data augmentation (Models 1, 2, 3, and 4), augmentation layers are incorporated before the base model. These layers apply random transformations such as horizontal flips, rotations, and zooms to enhance model robustness and generalization by artificially increasing the diversity of the training data.

## Model Checkpointing

To preserve the best-performing model during training, a checkpointing mechanism is employed. This saves the model weights whenever the validation accuracy improves, storing them in a specified file path (e.g., in HDF5 format). This ensures that the optimal model can be reloaded later without retraining.

## Usage

1. **Open the Notebook**:
   Launch the Colab notebook in Google Colab or a local Jupyter environment.

2. **Run the Cells**:
   Execute the cells sequentially to:
   - Verify TensorFlow version and GPU availability.
   - Import helper functions and the dataset.
   - Prepare training and test datasets using TensorFlow's image dataset utility.
   - Build, compile, and train models with the Functional API, incorporating augmentation and checkpointing as needed.

3. **Extend for Other Models**:
   Adapt the base model, augmentation settings, and trainable layers for Models 1–4. Adjust dataset sizes (e.g., 1% or 100%) by subsetting the training data accordingly.

## Helper Functions

From `helper_functions.py`:
- `create_tensorboard_callback`: Logs training metrics for TensorBoard visualization.
- `plot_loss_curves`: Plots training and validation loss/accuracy curves.
- `unzip_data`: Extracts the dataset zip file.
- `walk_through_dir`: Displays the dataset directory structure.

## Notes

- GPU usage is assumed but optional; the notebook runs on CPU if no GPU is detected.
- Validation is limited to 25% of the test data for faster experimentation.
- Extend the notebook with model evaluation and visualization steps to analyze results.


Map: Applies a preprocessing function to each data element.
Batch: Combines individual data elements into batches of a specified size.
Prefetch: Optimizes the data pipeline by loading the next batch while the current batch is being processed.

The Xception model is loaded with pre-trained weights.
The top layers of the Xception model are removed.
A Global Average Pooling layer is added to reduce the feature map.
A fully connected Dense layer with softmax activation is added for classification.
The final model is created by connecting the input of the base model to the output of the new dense layer.


@ONLINE {tfflowers,
author = "The TensorFlow Team",
title = "Flowers",
month = "jan",
year = "2019",
url = "http://download.tensorflow.org/example_images/flower_photos.tgz" }
