# ImageClassifierDSND

## 1. Installations   
importlib, pandas, numpy, matplotlib.pyplot, seaborn, collections, PIL, torch, torchvision, libraries are required

## 2. Project Motivation   

    Implement an image classification application using a deep learning model on a dataset of images and use the trained model to classify    new images. This project is part of Udacity Data Scientist term 1 Nanodegree. 

    I trained an image classifier to recognize different species of flowers using PyTorch and NN with GPU. 

    The project is broken down into multiple steps:

      - Load and preprocess the image dataset
      - Train the image classifier on your dataset
      - Use the trained classifier to predict image content

## 3. File Descriptions  
Image Classifier Project.html and Image Classifier Project.ipynb contain all the code in a html and Jupyter format respectively. 
predict.py, train.py, utils.py works to use the classifier with commands. 


### The command line application 

#### Specifications
The project include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. 
  
Basic usage: python train.py data_directory . 
Prints out training loss, validation loss, and validation accuracy as the network trains . 
  
Options:  
* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory . 
* Choose architecture: python train.py data_dir --arch "vgg16" . 
* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 . 
* Use GPU for training: python train.py data_dir --gpu . 

* Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.  
  
Basic usage: python predict.py /path/to/image checkpoint . 
  
Options:  
* Return top KK most likely classes: python predict.py input checkpoint --top_k 3 . 
* Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json .  
* Use GPU for inference: python predict.py input checkpoint --gpu .  

## 4. Licensing, Authors, Acknowledgements, etc.  
MIT
