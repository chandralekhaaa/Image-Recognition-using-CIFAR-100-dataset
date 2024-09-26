# Image-Recognition-using-CIFAR-100-dataset

Aim of the project:<br/> 
To classify images from the CIFAR-100 dataset into respective super classes.

Proposed Solution:<br/>
Utilizing a Convolutional Neural Network (CNN) optimized for accuracy to classify images from the CIFAR-100 dataset.

Dataset:<br/>
The CIFAR-100 dataset was used. CIFAR-100 is a subset of the Tiny Images dataset. It consists of 60,000 32×32 color images. The dataset contains 100 classes grouped into 20 superclasses. Each class includes 600 images, divided into 500 training images and 100 testing images. For example, some superclasses include flowers, insects, trees, etc. Examples of classes are spider, orchid, and willow tree.

Implementation:
<ul>
  <li>Data selection: The CIFAR-100 dataset was chosen due to its diverse range of classes and real-world complexity.</li>
  <li>Data exploration: Key statistics of the data were calculated, and sample images were displayed for better understanding.</li>
  <li>Model building and training: A CNN model was built and trained using TensorFlow and Keras. The model was trained for 20 epochs.</li>
  <li>Hyperparameter tuning: Keras Tuner was used to optimize hyperparameters.</li>
  <li>Model testing and evaluation: The tuned model was tested on the test dataset. The accuracy, confusion matrix, and classification report were displayed.</li>
  <li>Transfer learning: A MobileNetV2 model, pre-trained on ImageNet, was fine-tuned on the CIFAR-100 dataset. The model was trained for 30 epochs.</li>
</ul>

Conclusion and future work:
<ul>
  <li>The CNN model achieved a training accuracy of 81% and testing accuracy of 51%.</li>
  <li>If the CNN model were trained to classify into classes instead of super classes, it could have given better results.</li>
  <li>The transer learning did not yield favorable results.</li>
  <li>Future work could focus on developing a CNN model to classify based on classes instead of super classes.</li>
</ul>

References:<br/>
CIFAR-100 dataset - https://huggingface.co/datasets/uoft-cs/cifar100"/

