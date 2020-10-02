#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


# Creating meaningful text names for the various directories housing our data #
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[17]:


# DONE: Define transforms for the training, validation, and testing sets
# DEFINE TRANSFORMS & LOAD IMAGES INTO DATASETS USING THESE TRANSFORMS
                                         # TRAINING DATA....
train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0), # Randomly change brightness/ contrast / saturation of an image.
    transforms.RandomHorizontalFlip(p=0.3),                                # Horiz. flip with prob. indicated
    transforms.RandomVerticalFlip(p=0.3),                                  # Vert. flip with prob. indicated
    transforms.RandomRotation((0, 360)),                                   # Rotate the image by angle between 0 & 360 degrees.
    transforms.RandomResizedCrop(224),                                     # Random crop down to size = (224*224)
    transforms.ToTensor(),                                               # tfm to a tensor...
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],                     # ??? these means & std's push my numbers out of scale 0..1?!
      #                   std=[0.229, 0.224, 0.225])                       # eg.  !!  (0-0.485)/0.229 = -2.11 !!!!!!
    ])                    # ...tfm tensor with means & std devns to normalise each channel of the input
# Normalising helps keep the network weights near zero which in turn makes backpropagation more stable. 
# Without normalization, networks will tend to fail to learn.  

train_set = datasets.ImageFolder(train_dir, transform=train_transforms) 
# gets data from training dir to be processed with the transforms 'train_transforms'
# (images are auto-labelled with the name of their directory)

                                        # VALIDATION DATA....
valid_transforms = transforms.Compose([
    transforms.Resize(255),                                                # Resize to  (255*255)
    transforms.CenterCrop(224),                                            # Centre crop to size = (224*224)
    transforms.ToTensor(),                                                 # transform to a tensor..
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],                     # ??? these means & std's push my numbers out of scale 0..1?!
     #                    std=[0.229, 0.224, 0.225])                       # eg.  !!  (0-0.485)/0.229 = -2.11 !!!!!!
    ])
       
valid_set = datasets.ImageFolder(valid_dir, transform=valid_transforms) 
# gets data from validation dir to be processed with the transforms 'valid_transforms'
    
                                       # TESTING DATA....
test_transforms = transforms.Compose([
    transforms.Resize(255),                                                # Resize to  (255*255)
    transforms.CenterCrop(224),                                            # Centre crop to size = (224*224)
    transforms.ToTensor(),                                                 # transform to a tensor...
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],                     # ??? these means & std's push my numbers out of scale 0..1?!
   #                      std=[0.229, 0.224, 0.225])                       # eg.  !!  (0-0.485)/0.229 = -2.11 !!!!!!
    ])
    
test_set = datasets.ImageFolder(test_dir, transform=test_transforms) 
# gets data from test dir to be processed with the transforms 'test_transforms'     
##  ---------------------------------------------------   
    
# DONE: LOAD THE DATASETS INTO DATALOADERS with ImageFolder
trainloader = torch.utils.data.DataLoader(train_set, batch_size=46, shuffle=True) # randomly shuffles for every epoch
# pass train_set to DataLoader which returns batches of images & corresponding labels - for TRAINING
    
validationloader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True) # randomly shuffles for every epoch
# pass valid_set to DataLoader which returns batches of images & corresponding labels - for VALIDATION

testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True) # randomly shuffles for every epoch
# pass test_set to DataLoader which returns batches of images & corresponding labels - for TESTING
 
'''    
#_______________  LOOK AT SOME IMAGES TO SEE THAT THE LOADERS ARE OK ....  ________________   
##### Printing out all 64 images in the first batch - 
data_iter = iter(trainloader)             # make the loader (train/validate or test) iterable
images, labels = next(data_iter)          # batches of images & labels

plt.figure(figsize = [17, 13]) 
# creates new Fig, 14 inches by 14 inches (overall size for entire set of plots)
## loop through the batch & view all the images ##
n=0
for image in images:
    pic = image.permute(1, 2, 0)  # change the order dim[1] , foll'd by dim[2 then dim[0] for matplotlib to accept it
    # matplotlib requires RGB numbers to be in band 0..1 while Normalisation created the numbers from -1...1
    # the RGB numbers need to be rescaled to allow matplot lib to print the images
    #pic[0] = 0.5 + (pic[0] * 0.5)   # rescale  RGB channel data for matplotlib (to now fall in range 0..1)
    #pic[1] = 0.5 + (pic[1] * 0.5)   # rescale  RGB channel data for matplotlib (to now fall in range 0..1)
    print(image)
                
    plt.subplot(8, 8, n+1)        # divide the fig into an 8 by 8 (# Note: slot index numbers start at 1 (not 0). )
    plt.imshow(pic)
    n += 1
if n != 64:
    print("error")            # id if the no. images == that expected from the batch 


## ??????I have data - but I am unable to normalise...
### the values go from 0..1 to crazy numbers?????  Then matplotlib can't view them?!
'''


    


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


## load in a mapping from cat. no. to cat. name.  Held in file cat_to_name.json
## This provides a dict mapping the integer encoded categories to the actual names of the flowers
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name_dict = json.load(f)       # reading the dict into cat_to_name_dict  
'''  
    data_iter = iter(testloader)
    images, labels = next(data_iter)           # 'labels' is the next batch of labels
    for label in labels: 
        label_no = label.item()                # (to get the int '88' out of 'tensor[88]', use my_tensor.item())
        if str(label_no) in cat_to_name_dict:
            print(label_no, cat_to_name_dict[str(label_no)])         # prints each of the 64 labels
        else:
            print(label_no, "<<<no description found>>>")
# so i can see how to access the labels...........

no_batches = 0
no_labels = 0

#data_iter = iter(testloader)
for images, labels in testloader:
    no_batches += 1
    print("Label batch count : ", no_batches)
    for label in labels: 
        no_labels += 1
        print("Label count : ", no_labels, "  : ", end='')
        label_no = label.item()
        if str(label_no) in cat_to_name_dict:
            print(label_no, cat_to_name_dict[str(label_no)])         # prints each of the 64 labels
        else:
            print(label_no, "<<<no descrip found - not_a_flower!!!>>>")  
'''           
            
# Create a list of the 102 labels in label no. order
# (will be used as y_labels on the prob graph)

label_list = []

cat_to_name_dict = {int(k):str(v) for k,v in cat_to_name_dict.items()}   # make key an int so can be sorted properly
#print(cat_to_name_dict)

for key in sorted(cat_to_name_dict):                         # sort dict by key
    label_list.append(cat_to_name_dict[key])                 # append val to list
#print(label_list)                                            # correctly sorted now
label_list.insert(0,'not a flower>>>>>>>>')                 # add a {0 : 'not a flower'} option into the list
print(label_list, "\n\n")                                           # so, 103 final possible labels...get model sized correctly

# add 0 into dict
cat_to_name_dict[0] = 'not a flower>>>>>>>>'
print(cat_to_name_dict)
 


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[5]:


## DONE: BUILD & TRAIN MY NETWORK ##
from torchvision import models           # to access a pre-trained model
from torch import nn, optim              # for model classifier design & loss & optimiser choice
import time                              # for timing the training model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU access T-eed up
print("device is : ", device)

# I have needed to match the image normalisn used when the models were trained....
# (as per the means / std provided)  - which throws the rgb numbers out of range for matplotlib

# LOAD A PRE-TRAINED MODEL
model = models.resnet50(pretrained=True)     # has a 'features' component & a 'classifier' component
# think of trained networks as great feature detectors that can be used as input for simply feed-fwd classifiers
# this downloads the pretrained networks weights & biases as a starting pt for our model
# clearly, this will only serve us if our model can handle these


# FREEZE the 'FEATURES' PARAMETERS (we don't want these to get updated) & this speeds up training
for param in model.parameters():
    param.requires_grad = False    # turn off gradient calculation

# CREATE MY CLASSIFIER  (we want to REPLACE our 'classifier' with our own one ....)
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([            # naming my transforms & operations with OrderedDict()
    ('fc1', nn.Linear(2048, 1012)),                 # my model design of fully-connected layers
    ('relu1', nn.ReLU()),                            # starting with orig classifier's 'in_features=25088'?????too slow
    ('dropout1', nn.Dropout(p=0.2)),
    ('fc2', nn.Linear(1012, 512)),                 # my model design of fully-connected layers
    ('relu2', nn.ReLU()),                            # starting with orig classifier's 'in_features=25088'?????too slow
    ('dropout2', nn.Dropout(p=0.2)),
    ('fc3', nn.Linear(512, 103)),                  # 103 = no. of possible flower names (incl 1 for 'not a flower')
    ('output', nn.LogSoftmax(dim=1)),]))
model.fc = classifier                               # replace existing 'classifier' denoted "fc" in model defn, with mine

# select loss type; # select optimiser & learn rate
criterion = nn.NLLLoss()  
my_learn_rate = 0.15
optimizer = optim.SGD(model.fc.parameters(), lr=my_learn_rate)    
     
##  Training the model  ##
train_losses = []
val_losses = []
epochs = 15
model.to(device)                                        # move the processing to the GPU (if avail)
start = time.time()                                     # start a Training Loop timer

for e in range(epochs):
    
    running_loss = 0
    no_batches = 0
    print("Now training....")
    print("Starting Epoch no: {}".format(e+1))
    print("Learn rate : ",my_learn_rate)
    
    for images, labels in trainloader:                 # each step in the for loop == all images in a batch of training data
        optimizer.zero_grad()                          # Clear the gradients for each batch (else they accumulate)
        
        ### Training pass  ###  
        images, labels = images.to(device), labels.to(device)   # ensure images & labels are also moved to GPU
        logps = model(images)               # 1..4 FORWARD PASS (from nn) to get log probs (logsoftmax used)
        loss = criterion(logps, labels)     # 2..4 CALCULATE LOSS (from nn) used log probabilities
        loss.backward()                     # 3..4 BACKWARD PASS using our loss (this calcs gradients)  (from nn)
        optimizer.step()                    # 4..4 UPDATE WEIGHTS & BIASES based on the gradients & learn rate chosen (from optim)
        
        no_batches += 1
        running_loss += loss.item()                    # update the running loss by adding loss it
        
        if no_batches % 10 == 0:
            print("Training loss after {:3d} batches in epoch {}, is {:.3f}".format(no_batches, e+1, running_loss/no_batches))
        # at this stage ALL IMAGES (all batches) IN TRAINLOADER have been run through...
                
    else:       #  If no bk command in the for loop, then the else part will be called.
        print(f"Training loss: {e+1, running_loss/len(trainloader)}") 
        
        val_loss = 0
        accuracy = 0
        no_batches = 0
        print("Now validating....")
        
        with torch.no_grad():                            # turn off gradients for validation for speed/memory
            for images, labels in validationloader:      # for all 13 batches in the validation data
                images, labels = images.to(device), labels.to(device)   # ensure images & labels are also moved to GPU
                ### Validation pass  ###  
                logps = model(images)                   # FORWARD PASS to get log probs (logsoftmax used)
        
                no_batches += 1
                val_loss += criterion(logps, labels)    # calc & accumulate losses 
                ps = torch.exp(logps)                   # get the probabilities back from the log probs
        
                top_k, top_class = ps.topk(1, dim=1)    # get the predicted class of the highest probability into top_class
                # create a 'byte tensor' called equals where (top_class == labels)  {labels had shape (64) resized to be same as top_class shape}
                equals = top_class == labels.view(*top_class.shape)     # 'True' if predicted class matches true class
                accuracy += torch.mean(equals.type(torch.FloatTensor))  # mean does not work on a byte tensor...make equals a float tensor
                # torch.mean() returns a scalar tensor, to get the actual value as a float we'll need to do accuracy.item().
                print("Batch No. : {}".format(no_batches), "Test/validation Loss: {:.3f}".format(val_loss/no_batches),"Validation Accuracy: {:.2f}%".format(accuracy.item()*100/no_batches))
        
        train_losses.append(running_loss/len(trainloader))
        val_losses.append(val_loss/len(validationloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f} ".format(running_loss/len(trainloader)),
              "Test/validation Loss: {:.3f} ".format(val_loss/len(validationloader)),
              "Test/validation Accuracy: {:.2f}%".format(accuracy.item()*100/len(validationloader)))
        print(train_losses)
        print(val_losses)
        
    
stop = time.time()                                      # stop the Training Loop timer 
time_per_epoch = (stop - start)/epochs
print(f"Device is: {device}; Time per epoch: {time_per_epoch:.3f} seconds")  # print ave time/epoch 
time_per_batch = time_per_epoch/no_batches
print("Ave time per batch : ", time_per_batch, " in seconds.")



# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[9]:


# DONE: Do validation using the test set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU access T-eed up
print("device is : ", device)

# select loss type
criterion = nn.NLLLoss()  
       
##  Testing the model  with test data  ##
test_losses = []
model.to(device)                                        # move the processing to the GPU (if avail)
        
test_loss = 0
accuracy = 0
no_batches = 0
print("Now testing....")
        
with torch.no_grad():                      # turn off gradients for testing - for speed/memory
    for images, labels in testloader:      # for all <<819 batches>>  13 batches in the test data
        images, labels = images.to(device), labels.to(device)   # ensure images & labels are also moved to GPU
        ### Test pass  ###  
        logps = model(images)                   # FORWARD PASS to get log probs (logsoftmax used)
        
        no_batches += 1
        test_loss += criterion(logps, labels)    # calc & accumulate losses 
        ps = torch.exp(logps)                   # get the probabilities back from the log probs
        
        top_k, top_class = ps.topk(1, dim=1)    # get the predicted class of the highest probability into top_class()
        print("Top class no. predicted is : ", top_class)        # for testing vis - remove later...
        print("Top class's probability predicted is : ", top_k)  # for testing vis - remove later...
        print("Correct label: ", labels)                         # for testing vis - remove later...
        
        # create a 'byte tensor' called equals where (top_class == labels)  {labels had shape (64) resized to be same as top_class shape}
        equals = top_class == labels.view(*top_class.shape)     # 'True' if predicted class matches true class
        accuracy += torch.mean(equals.type(torch.FloatTensor))  # mean does not work on a byte tensor...make equals a float tensor
        # torch.mean() returns a scalar tensor, to get the actual value as a float we'll need to do accuracy.item().
        print("Batch No. : {}".format(no_batches), 
              "Test Loss: {:4.2f} ".format(test_loss/no_batches),
              "Test Accuracy: {:1.1f}%".format(accuracy.item()*100/no_batches))
        
        
        
        #added the next section in just to see if I can sanity check on a rolling set of data that gives good accuracy?
        
        if no_batches % 1 == 0:
            #image_path = images[0]
            #image_path.unsqueeze_(0)
            #logps = model(image_path)                        # FORWARD PASS to get log probs (logsoftmax used)
            #image_path.squeeze_(0) 
            
            #logps = model(images)
            #ps = torch.exp(logps)                            # get the probabilities back from the log probs
            top_k, top_class = ps.topk(5, dim=1)             # get the 5 highest predicted classes into tensor [top_class]
                                                             # & get the 5 highest prob'abilities' into tensor [top_k]
            y_labels = []                                    # get the labels for the 5 - for use on the y-axis of the prob graph
            top_class_list = top_class.tolist()              # top_class_list is like a list within a list...
            clean_top_class_list = top_class_list[0]                    # ...peeling the onion
            print("clean_top_class_list : ",clean_top_class_list)
        
            for num in clean_top_class_list:
                y_labels.append(cat_to_name_dict[num])            # y-labels now contains top 5 descriptions
            print("y_labels:   ", y_labels)    
            
            plt.figure(figsize = [8, 6])  # creates new Fig, 3" by 6"  (overall size for both plots)
                           # Remove the first dim (the batch that we added to get this through FWD PASS)
            print("Shape     ", image_path.shape)
            pic = image_path.permute(1, 2, 0)  # change the order dim[1], foll'd by dim[2] then dim[0] for matplotlib to accept it
    
            plt.subplot(1, 2, 1)          # first subplot: divide the fig into a 1 by 2 (# Note: slot index numbers start at 1 (not 0). )
            plt.imshow(pic)               # show pic
   
            plt.subplot(1, 2, 2)          # second subplot: divide the fig into a 1 by 2 (# & this is slot 2..2)
            plt.xticks(rotation=45)
    
            top_k_list = top_k.tolist()                         # [[top_k_list]] is like a list within a list...
            clean_top_k_list = top_k_list[0]                    # ...peeling the [onion]
            print("clean_top_k_list:   ", clean_top_k_list)
    
            plt.bar(y_labels, clean_top_k_list)
            plt.show()
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        
   
        
        
        
    print("Latest Test Loss: {:.2f}.. ".format(test_loss/len(testloader)), 
          "Latest Test Accuracy: {:.1f}%".format(accuracy.item()*100/len(testloader)))
    
    test_losses.append(test_loss/len(testloader))
    print(test_losses)
        


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[8]:


# Review the model before saving a checkpoint
print("The model for which I am considering saving a checkpoint, is : \n\n", model)
print("The associated state_dict, is held in model.state_dict()'\n\n")
print("The keys from model.state_dict() are: \n")
for key in model.state_dict():     # have a look at the model.state_dict() keys
    print(key)

print("\n\n The associated optimiser state, is held in optimiser.state_dict()'\n\n")
print("The keys from optimiser.state_dict() are: \n")
for key in optimizer.state_dict():     # have a look at the optimiser.state_dict() keys
    print(key)
# ________________________________________________________________________________
print("____________________________________________________________________")

#  DONE: SAVE THE CHECKPOINT  #
# When we load the weights & biases from the state_dict, we need to input these into a model 
# which is identical to that which created the w&b's in the first place.
# So, it makes sense to save a checkpoint which contains not only the w&b's but also the model architecture.


# 1..2 CREATE THE CHECKPOINT - { a dict of (the model arch. & the state_dict (containing the weights & biases)) }
# !!  Ensure you setup the checkpoint model arch. to mirror that of the model you wish to save !!
checkpoint = {'input_size' : 2048,
              'fc1' : 1012,
              'fc2' : 512,
              'fc3' : 103,
              'output_size' : 103,
              'state_dict' : model.state_dict(),
              'opt_state_dict' : optimizer.state_dict(),
              'no_epochs' : 15,
              'labels_dict' : cat_to_name_dict,
              'criterion' : 'nn.NLLLoss()',
              'my_learn_rate' : 0.15
             }

# 2..2 SAVE THE CHECKPOINT to file checkpoint1.pth in current dir
torch.save(checkpoint, './checkpoint1.pth')
print("saved...")


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[5]:


# DONE: Write a function that loads a checkpoint and rebuilds the model
import torch
from torchvision import models           # to access a pre-trained model
from torch import nn, optim              # for model classifier design & loss & optimiser choice

def load_checkpoint(file_path):
    '''
    Takes the filepath of a checkpoint & loads 
    - the MODEL that created the checkpoint, &
    - the ASSOCIATED (WEIGHTS & BIASES) from the model's state_dict
    '''
    if torch.cuda.is_available():                        # to enable loading of the checkpoint when GPU not available
        map_location = lambda storage, loc: storage.cuda()   
    else:
        map_location='cpu'

    checkpoint = torch.load(file_path, map_location=map_location)  # 1 - load 'chkpoint' dict from file, into memory
                                                                   # (look in appropriate map_location)
    # get my model
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False    # turn off gradient calculation
    # rebuild my classifier
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([           
    ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['fc1'])),                
    ('relu1', nn.ReLU()),                          
    ('dropout1', nn.Dropout(p=0.2)),
    ('fc2', nn.Linear(checkpoint['fc1'], checkpoint['fc2'])),                
    ('relu2', nn.ReLU()),                           
    ('dropout2', nn.Dropout(p=0.2)),
    ('fc3', nn.Linear(checkpoint['fc2'], checkpoint['fc3'])),                  
    ('output', nn.LogSoftmax(dim=1)),])) 
    
    model.fc = classifier         # add classifier into the model
    
    model.load_state_dict(checkpoint['state_dict'])          # 3 - load the state_dict into the model's state_dict
        
    epochs = checkpoint['no_epochs']                         # 4 - load no epochs into memory
    labels_dict = checkpoint['labels_dict']                  # 4 - load labels_dict into memory
    optimizer_state_dict = checkpoint['opt_state_dict']      # 4 - load optimizer_state_dict into memory
    criterion = checkpoint['criterion']
    my_learn_rate = checkpoint['my_learn_rate']
                
    return model, epochs, criterion, my_learn_rate, labels_dict, optimizer_state_dict


# load the checkpoint & have a look to see that it has correctly loaded
model, epochs, criterion, my_learn_rate, labels_dict, optimizer_state_dict = load_checkpoint('./checkpoint1.pth')
print(model)
print(epochs)
print(criterion)        
print(my_learn_rate)        
print(labels_dict)
print(optimizer_state_dict)
    


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[9]:


# TODO: Process a PIL image for use in a PyTorch model

from PIL import Image
import numpy as np




def process_image(image):
    # Scales, crops, and normalizes a PIL image for a PyTorch model,
    # returns an Numpy array
    
    size = 256, 256
    image.thumbnail(size)                       # resize with shortest side 256, preserving existing aspect ratio
    
    width, height = image.size                     # Get dimensions & centre crop to 224*224
    req_width = 224                                # co-ords are based on a cartn. plain with top left = (0, 0)
    req_height = 224                               # calculate the numbers for left/top/right/bottom
    left = (width - req_width)/2
    top = (height - req_height)/2
    right = (left + req_width)
    bottom = (top + req_height)

    # Crop the center of the image                          # 'left' = top left x; 'top' = top left y;
    image_cropped = image.crop((left, top, right, bottom))  #  'right' = bottom right x; 'bottom' = bottom right y
  
    # Convert PIL to a NUMPY array
    np_image = np.array(image_cropped)
    
    return np_image

    # Normalise the RGB channels
   

# Get image  'im'

im = Image.open("Cat_01.jpg")
im.show()                            # simply does not display anything?!


# Process the image
processed_image = process_image(im)

print(processed_image.shape)            # shape (224, 224, 3)

means = np.array([0.485, 0.456, 0.406])  # shape (3,)
print(means.shape)
std = np.array([0.229, 0.224, 0.225])      # shape (3,)
print(std.shape)


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[ ]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[86]:


# TODO: Implement the code to predict the class from an image file
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # for visualisations   
get_ipython().run_line_magic('matplotlib', 'inline')

def predict(image_path, model, labels_dict, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    
    with torch.no_grad():                            # turn off gradients for prediction - for speed/memory
        logps = model(image_path)                    # FORWARD PASS to get log probs (logsoftmax used)
    ps = torch.exp(logps)                            # get the probabilities back from the log probs
    print(ps)
        
    top_k, top_class = ps.topk(5, dim=1)             # get the 5 highest predicted classes into tensor [top_class]
                                                     # & get the 5 highest prob'abilities' into tensor [top_k]
    y_labels = []                                    # get the labels for the 5 - for use on the y-axis of the prob graph
    top_class_list = top_class.tolist()              # top_class_list is like a list within a list...
    clean_top_class_list = top_class_list[0]                    # ...peeling the onion
    print("clean_top_class_list : ",clean_top_class_list)
        
    for num in clean_top_class_list:
        y_labels.append(labels_dict[num])            # y-labels now contains top 5 descriptions
    print("y_labels:   ", y_labels)    
            
    plt.figure(figsize = [8, 6])  # creates new Fig, 3" by 6"  (overall size for both plots)
    image_path.squeeze_(0)                # Remove the first dim (the batch that we added to get this through FWD PASS)
    print("Shape     ", image_path.shape)
    pic = image_path.permute(1, 2, 0)  # change the order dim[1], foll'd by dim[2] then dim[0] for matplotlib to accept it
    
    plt.subplot(1, 2, 1)          # first subplot: divide the fig into a 1 by 2 (# Note: slot index numbers start at 1 (not 0). )
    plt.imshow(pic)               # show pic
   
    plt.subplot(1, 2, 2)          # second subplot: divide the fig into a 1 by 2 (# & this is slot 2..2)
    plt.xticks(rotation=45)
    
    top_k_list = top_k.tolist()                         # [[top_k_list]] is like a list within a list...
    clean_top_k_list = top_k_list[0]                    # ...peeling the [onion]
    print("clean_top_k_list:   ", clean_top_k_list)
    
    plt.bar(y_labels, clean_top_k_list)
    plt.show()
    
    return clean_top_k_list, clean_top_class_list
    

# Get an image in a tensor format - say from the validation dir
valid_transforms = transforms.Compose([
    transforms.Resize(255),                                                # Resize to  (255*255)
    transforms.CenterCrop(224),                                            # Centre crop to size = (224*224)
    transforms.ToTensor(),                                                 # transform to a tensor..
    ])
       
valid_set = datasets.ImageFolder('flowers/valid', transform=valid_transforms) 
# gets data from validation dir to be processed with the transforms 'valid_transforms'

validationloader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True) # randomly shuffles for every epoch
# pass valid_set to DataLoader which returns batches of images & corresponding labels - for VALIDATION

data_iter = iter(validationloader)        # make the loader iterable
images, labels = next(data_iter)          # get first batch of images & labels
chosen_image = images[63]                  # 6th image in the batch
print("image shape before unsqueezed: ", chosen_image.shape)      # shape is (3, 244, 244)
chosen_image.unsqueeze_(0)                # adding a batch dimension else trps up on 'stride size'
print("image shape when sent to fn :", chosen_image.shape)                 # shape is (1, 3, 244, 244)
        
       
probs, classes = predict(chosen_image, model, labels_dict)  # arg1 must be a tensor
print("Probs: ", probs)
print("Classes :", classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[ ]:


# TODO: Display an image along with the top 5 classes


# In[ ]:


# DONE: Sanity test using the data str from the test loader .............MESSS WITH THIS.....

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU access T-eed up
print("device is : ", device)

# select loss type
criterion = nn.NLLLoss()  
       
##  Testing the model  with test data  ##
test_losses = []
model.to(device)                                        # move the processing to the GPU (if avail)
        
test_loss = 0
accuracy = 0
no_batches = 0
print("Now testing....")
        
with torch.no_grad():                      # turn off gradients for testing - for speed/memory
    for images, labels in testloader:      # for all <<819 batches>>  13 batches in the test data
        images, labels = images.to(device), labels.to(device)   # ensure images & labels are also moved to GPU
        ### Test pass  ###  
        
        print("Images.shape :", images.shape)
        print("Images[0].shape :", images[0].shape)
        image = images[0].unsqueeze_(0)     # adding a batch dimension else trips up on 'stride size'
        print("Image.shape after unsqueeze :", images[0].shape)
        logps = model(image)    # FORWARD PASS on the first batch ie 1 image as batch size is 1
        
        no_batches += 1
        test_loss += criterion(logps, labels)    # calc & accumulate losses 
        ps = torch.exp(logps)                   # get the probabilities back from the log probs
        
        top_k, top_class = ps.topk(1, dim=1)    # get the predicted class of the highest probability into top_class()
        print("Top class no. predicted is : ", top_class)        # for testing vis - remove later...
        print("Top class's probability predicted is : ", top_k)  # for testing vis - remove later...
        print("Correct label: ", labels)                         # for testing vis - remove later...
        
        # create a 'byte tensor' called equals where (top_class == labels)  {labels had shape (64) resized to be same as top_class shape}
        equals = top_class == labels.view(*top_class.shape)     # 'True' if predicted class matches true class
        accuracy += torch.mean(equals.type(torch.FloatTensor))  # mean does not work on a byte tensor...make equals a float tensor
        # torch.mean() returns a scalar tensor, to get the actual value as a float we'll need to do accuracy.item().
        print("Batch No. : {}".format(no_batches), 
              "Test Loss: {:4.2f} ".format(test_loss/no_batches),
              "Test Accuracy: {:1.1f}%".format(accuracy.item()*100/no_batches))
        
        
        
        #added the next section in just to see if I can sanity check on a rolling set of data that gives good accuracy?
        
        if no_batches % 1 == 0:
            #image_path = images[0]
            #image_path.unsqueeze_(0)
            #logps = model(image_path)                        # FORWARD PASS to get log probs (logsoftmax used)
            #image_path.squeeze_(0) 
            
            #logps = model(images)
            #ps = torch.exp(logps)                            # get the probabilities back from the log probs
            top_k, top_class = ps.topk(5, dim=1)             # get the 5 highest predicted classes into tensor [top_class]
                                                             # & get the 5 highest prob'abilities' into tensor [top_k]
            y_labels = []                                    # get the labels for the 5 - for use on the y-axis of the prob graph
            top_class_list = top_class.tolist()              # top_class_list is like a list within a list...
            clean_top_class_list = top_class_list[0]                    # ...peeling the onion
            print("clean_top_class_list : ",clean_top_class_list)
        
            for num in clean_top_class_list:
                y_labels.append(cat_to_name_dict[num])            # y-labels now contains top 5 descriptions
            print("y_labels:   ", y_labels)    
            
            plt.figure(figsize = [8, 6])  # creates new Fig, 3" by 6"  (overall size for both plots)
                           # Remove the first dim (the batch that we added to get this through FWD PASS)
            print("Shape     ", image_path.shape)
            pic = image_path.permute(1, 2, 0)  # change the order dim[1], foll'd by dim[2] then dim[0] for matplotlib to accept it
    
            plt.subplot(1, 2, 1)          # first subplot: divide the fig into a 1 by 2 (# Note: slot index numbers start at 1 (not 0). )
            plt.imshow(pic)               # show pic
   
            plt.subplot(1, 2, 2)          # second subplot: divide the fig into a 1 by 2 (# & this is slot 2..2)
            plt.xticks(rotation=45)
    
            top_k_list = top_k.tolist()                         # [[top_k_list]] is like a list within a list...
            clean_top_k_list = top_k_list[0]                    # ...peeling the [onion]
            print("clean_top_k_list:   ", clean_top_k_list)
    
            plt.bar(y_labels, clean_top_k_list)
            plt.show()
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        
   
        
        
        
    print("Latest Test Loss: {:.2f}.. ".format(test_loss/len(testloader)), 
          "Latest Test Accuracy: {:.1f}%".format(accuracy.item()*100/len(testloader)))
    
    test_losses.append(test_loss/len(testloader))
    print(test_losses)
        


# In[ ]:




