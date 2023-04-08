# DeepLearning_Challenges

# Story Ending Classification Challenge

Dataset
Training set: 52665 stories
Testing set: 1571 stories

Code Structure

1. Load the pre-trained BERT model for masked language modelling (MLM)
2. Fine-tune the BERT model on the flower dataset using MLM
3. Transfer the weights of the fine-tuned BERT model to a BERT model for multiple-choice tasks
4. Fine-tune the BERT model for multiple-choice tasks on the flower dataset
5. Save the model for future use

# Flower Image Classification Challenge

Dataset
Training set: 2040 images
Testing set: 6149 images

In order to solve this problem, the following pre-trained models were experimented with:

1. ResNet-50: A popular convolutional neural network (CNN) architecture with 50 layers, known for its high performance on image recognition tasks. I fine-tuned this model by replacing the final layer with a custom linear layer and a softmax activation function to predict the class probabilities of the input images.

2. VGG-19: Another popular CNN architecture with 19 layers, VGG-19 is known for its simplicity and high performance on image recognition tasks. I followed a fine-tuning approach as with ResNet-50, replacing the final layer with a custom linear layer and softmax activation function.

3. DenseNet-161: A deep CNN architecture with 161 layers, DenseNet-161 is characterized by its dense connections between layers. I fine-tuned this model by unfreezing some layers and adding a custom classifier to predict the class probabilities of the input images.
   Data augmentation techniques, such as random rotations, random resized crops, and random horizontal flips, were used to improve the model's generalization capabilities. I also normalized the input images with the appropriate mean and standard deviation values for each pre-trained model.

I used different optimization techniques and learning rate scheduling to train the models. In particular, I experimented with AdamW, RMSprop, and SGD optimizers along with OneCycleLR and ReduceLROnPlateau learning rate schedulers.

Finally, the training and validation loss curves were plotted to visualize the model's performance over time.
