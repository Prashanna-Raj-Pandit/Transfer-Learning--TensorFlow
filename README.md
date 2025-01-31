# Transfer-Learning--TensorFlow

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
