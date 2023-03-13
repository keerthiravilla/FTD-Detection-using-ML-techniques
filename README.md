# Detection of Frontotemporal Dementia by Learning Few Training Samples
# Background 
Convolutional neural networks (CNNs) achieve the high classification accuracy for detecting
frontotemporal dementia with a large number of training samples based on magnetic resonance
imaging (MRI) scans, but they didn‘t achieve good diagnostic accuracy with few training samples.
One important reason is that in the medical domain, the acquisition of large number of trainng samples
is quite hard and complicated due to the patients‘ privacy concerns. Recently developed a few-shot learning methodology that
deals with the data insufficiency problem. Few-shot learning methodology proposes the strategies
through which we resolve the problem of data insufficiency and achieve the classification performance
as same as with a large number of training samples. We investigate the detection of frontotemporal
dementia using only a few MRI scans for training.

# Methods: 
We utilized the transfer learning and few-shot learning methodologies to overcome the
problem of a few available training samples. Firstly, we created the feature extraction model that is
trained on the large ADNI dataset (a total of 662 samples). This developed model is the convolutional
neural network that learns feature representations based on ADNI MRI scans. Furthermore, we
transfer the representations learned by the feature extraction model to the model that is trained on
the small FTD dataset (a total of 279 data samples) by following a model perspective-based embedding
learning methodology of few-shot learning.

# Results: 
We developed the CNN models utilizing the transfer learning methods that learn the optimal
feature representations. The CNN model with the fine-tuning method based on the ADNI dataset
achieves the Alzheimer's disease classification accuracy of 0.97. Secondly, we achieved the classification
accuracy of FTD disease with only 20 training samples of 0.63. As we increase the training samples
up to 40 we achieved the FTD diagnostic accuracy of 0.75.
