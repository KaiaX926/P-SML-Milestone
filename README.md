# P-MNIST-milestone

The MNIST database of handwritten digits is one of the most commonly used dataset for training various image processing systems and machine learning algorithms. MNIST has a training set of 60,000 examples, and a test set of 10,000 examples. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.
MNIST is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. The original black and white (bilevel) images from NIST were size normalized. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. The images were centered in a 28 × 28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28 × 28 field.

In the first two questions completed before, I explored the simple learners, including KNN, SVM, and Decision tree. Some of them already have great performance. In the second phase, I used machine-learning algorithms to predict the results. Cross entropy loss and accuracy of the model are considered throughout the training. Using different tricks like dropout, batch normalization, and momentum, I found the model that will outperform the simple learners.

## Findings
Three main observations are obtained during the whole process based on the training and the model behaviors. Detailed explanation refers to the reports. Detailed explanations and comparisons are commented on in the code.
1. The seed will influence the behavior of models.
2. Convolutional layers do improve the behaviors of the neuron network.
3. Momentum and learning rate is crucial to the model performance. 
4. Quality of data set  is important to the model performance.

## Files
1. report: The detailed explanation of the training process and answers to the questions required.
2. plots_generator.pdf: The original code of this pdf has some issues. The loss of the validation set is not on a similar scale since I forget to times the batch number when calculating the loss of the validation set. However, this will not influence the trends and the visualization of filters. Also, some accuracy is not printed directly. I added this file because I don't have enough time to re-run through the updated code to obtain all the correct numbers. This pdf will help understand the result visually.
3. 5241-milestone-0508.ipynb: The final version of code. Although not all the plots are generated, you can run through them to get the answers. 

## Nets
Below is Nets that I considered with proper activation functions:

**Net1**(\
  (fc1): Linear(in_features=784, out_features=100, bias=True)\
  (output): Linear(in_features=100, out_features=10, bias=True)\
)

**Net2**(\
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))\
  (out): Linear(in_features=1176, out_features=10, bias=True)\
)

**Net3**(\
  (conv1): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))\
  (conv2): Conv2d(16, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))\
  (batch1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\
  (conv4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\
  (batch2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
  (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\
  (batch3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
  (fc1): Linear(in_features=2304, out_features=512, bias=True)\
  (fc2): Linear(in_features=512, out_features=128, bias=True)\
  (fc3): Linear(in_features=128, out_features=10, bias=True)\
)

**Net4**(\
  (conv1): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))\
  (conv2): Conv2d(16, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))\
  (batch1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\
  (conv4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\
  (batch2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
  (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\
  (batch3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
  (fc1): Linear(in_features=2304, out_features=512, bias=True)\
  (fc2): Linear(in_features=512, out_features=128, bias=True)\
  (fc3): Linear(in_features=128, out_features=19, bias=True)\
)

**Net5**(\
  (conv1): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))\
  (conv2): Conv2d(16, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))\
  (batch1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\
  (conv4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\
  (batch2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
  (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\
  (batch3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
  (fc1): Linear(in_features=2304, out_features=512, bias=True)\
  (fc2): Linear(in_features=512, out_features=128, bias=True)\
  (fc3): Linear(in_features=128, out_features=19, bias=True)\
  (dropout): Dropout(p=0.25, inplace=False)\
)
