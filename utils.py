


import pandas as pd
import numpy as np
import json
import importlib

from collections import OrderedDict
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def data_loaders(data_dir = 'flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size = 128)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 128)
    
    return trainloader, validloader, testloader, train_data


def build_network(arch = "vgg16"):   
    
    model = getattr(importlib.import_module('torchvision.models'), arch)(pretrained=True)

    input_size = int(str(model.classifier).split("in_features=")[1].split(",")[0])
    for param in model.parameters():
        param.requires_grad = False
    #input_shape = model.output_shape[1]
    Classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, 512)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p = 0.4)),
        ('fc2', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
        
    model.classifier = Classifier
    print(model)
    return model, input_size

        
        
def trainer(trainloader, validloader, model, epochs = 5, steps = 0,  learnrate = 0.001, print_every = 5, gpu = 'gpu'):

    running_loss = 0
    
    criterion = nn.NLLLoss()
    print(model.classifier.parameters())
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    
    if gpu == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            print('cuda is not available, please check it')
    else:
        device = 'cpu'
    print (device)
    model.to(device);

    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    return model, optimizer
                
    
    
def save_checkpoint(model, train_data, optimizer, input_size, output_size, hidden_layers, learn_rate, epochs, steps = 0):
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'model': model,
                  'input_size': input_size ,
                  'output_size': output_size,
                  'hidden_layers':  hidden_layers,
                  'state_dict': model.state_dict(), 
                  'class_to_idx' : model.class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  'learn_rate': learn_rate, 
                  'epochs': epochs,
                  'steps': 0
                 }

    torch.save(checkpoint, 'checkpoint.pth')
    
def category(category):
    with open(category, 'r') as json_file:
        cat_to_name = json.load(json_file)
    return cat_to_name
def load_checkpoint(path="checkpoint.pt"):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learn_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
 
def resize (img, size):
    w, h = img.size
    
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh))
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh))

    return img.resize(size[::-1])

def center_corp(img, size):
    w, h = img.size
    left = (w - size) / 2
    top = (h - size) / 2
    right = (w + size) / 2
    bottom =(h + size) / 2
    
    return img.crop((left, top, right, bottom))

def normalize(img, mean, std):
    img = np.array(img) / 255
    return (img - mean) / std
    

        
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image).copy()
    img = resize(img, 256)
    img = center_corp(img, 224)
    img = normalize(img,   mean = np.array([0.485, 0.456, 0.406]) ,std = np.array([0.229, 0.224, 0.225]))
    return  img.transpose((2,0,1))
          


def predict(image_path, model, cat_to_name, device = 'cuda', topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if device == "gpu":
        device = "cuda"
    im = process_image(image_path)
    im =torch.from_numpy(np.array(im)).float()
    im.unsqueeze_(0)
    im = im.to(device)
    output = model.forward(im)
    probs, topk_class = torch.exp(output).topk(topk)
    probs = probs.tolist()[0]
    topk_class = topk_class.tolist()[0]
    idx_to_class = { v : k for k,v in model.class_to_idx.items()}
    topk_class = [idx_to_class[x] for x in topk_class]
    classes = [cat_to_name[x] for x in topk_class]
    
    print(probs, classes)
    
    table = '{:<10}{:<10}{}'
    print(table.format('', 'Classes', 'Probability'))
    for i, (class_, probs) in enumerate(zip(classes, probs)):
        print(table.format(i, class_, probs))
    return probs, classes
