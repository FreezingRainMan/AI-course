# PROGRAMMER: Michael Gaylard
# DATE CREATED: 8th October 2020                                 
# REVISED DATE: 
# PURPOSE: t_setup.py provide the following functions that are used by the script 'train.py         
#          These functions are:
#             get_input_args()                    : Retrieves Command Line Arguments from user & returns these (or the defaults), as the variable in_args                   #             check_command_line_arguments()      : Validates Command Line Arguments and exits if validation rules are broken
#             create_datasets()                   : Gets and loads the data file from the location as specified by the user (or the in_args default)
#             get_cat_to_name_dict()              : Loads the mapping (cat. no. to cat. name) of the test data into a dictionary
#             load_checkpoint()                   : Loads the checkpoint (Model & Optimizer) from the chosen file (or the in_args default)
#             train_model()                       : Trains the model using the input arguments from in_args (or the in_args defaults)
#             save_checkpoint()                   : Saves the model checkpoint after training - to a chosen file (or the in_args default)
#
# Import python modules
import argparse                                   # for getting inputs from the command line & subsequently parsing them
import os                                         # to perform validity check on file paths captured on the command - to check that they exist
import torch
from torchvision import datasets, transforms      # to create datasets
from torchvision import models                    # to access a pre-trained model
import torchvision                                # required to re-create the architecture
from torch import nn, optim                       # for model classifier design & loss & optimiser choice
import json                                       # to access the cat_to_name_dict info                                     
from time import time                             # for timing the training model

# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when they run the program from a terminal window. 
    This function uses Python's argparse module to created and define these command line arguments.
    If the user fails to provide some or all of the 6 arguments, then the default values are used for the missing arguments. 
    Command Line Arguments:
      1. Data directory path                                  any valid directory                                                 default 'flowers'        
      2. Model architecture as --arch < >                     one of ("densenet121", "alexnet", "resnet50", "resnet152")          default 'resnet50'             
      3. Learning rate as --learn < >                         possible values of (0 < value <= 1)                                 default 0.15
      4. Number of hidden units as --hidden_units < >         possible values of (0 < value)                                      default 1012
      5. Number of epochs as --epochs < >                     possible values of (1 < value <= 50)                                default 2
      6. Path for saving checkpoints as --save_dir < >        any valid path & filename ending in .pth                            default "./checkpoint100.pth" 
      7. GPU requested as --gpu                               n/a                                                                 flag defaulted to 'gpu not required'
      8. Train using an already-trained checkpoint file  --chk  no values to follow!  You'll be prompted for a filename.                      
    This function returns these arguments as a dictionary.
         Parameters:    None - simply using argparse module to create & store command line arguments
         Returns:       in_args() - a dictionary that stores the command line argument names & values 
    """
    # Instantiates an ArgumentParser object called 'parser'
    parser = argparse.ArgumentParser()
     
    # Creates POSITIONAL command line arguments, using add_argument() method from ArgumentParser 
    parser.add_argument("data_dir",       help = "Directory in currect folder to get the test data - (default 'flowers')", nargs = '?', type = str, default = 'flowers')
    #nargs = '?'  means num of arguments could be one or none - in which case the default is used
    #It is assumed that data in directories other than 'flowers' will have the same subdirectory structure otherwise the create_datasets() function will fail
    
    # Creates OPTIONAL command line arguments, using add_argument() method from ArgumentParser 
    parser.add_argument("--arch",         help = "Model architecture - (default 'resnet50')",             type = str, default = 'resnet50')
    
    parser.add_argument("--learn",        help = "Learning rate - (default  0.15)",                       type = float, default = 0.15)
    parser.add_argument("--hidden_units", help = "Number of hidden units - (default 1012)",               type = int, default = 1012)
    parser.add_argument("--epochs",       help = "Number of epochs - (default of 2)",                     type = int, default = 2)
    parser.add_argument("--save_dir",   help = "Path for saving the trained model checkpt - (default './checkpoint100.pth')", type = str, default = './checkpoint100.pth' )
    parser.add_argument("--gpu",          help = "Sets flag that gpu should be used",                      action='store_true') 
    parser.add_argument("--chk",          help = "Sets flag that a checkpoint file should be used",        action='store_true') 
    # If you use the flag --gpu then 'gpu' in the dictionary value will be True, otherwise it will be False 
    
    in_args = parser.parse_args() 
    # Returns the parsed argument collection created within this function 
    # this 'parsed argument collection' is a 'Namespace' of format Namespace(dir = 'mydir', arch = 'myarch', dogfile = 'mydogfile')
    in_args = vars(in_args)
    # Converts the Namespace into a dictionary such as {'arch' : 'vgg13', 'learn' : '0.4', 'gpu' : 'gpu'} 
    # for easier accessing later.  e.g. using in_args['dir'] to return 'mydir'
   
    return in_args        #  in_args is a dict containing pairs of {name of command line argument : value of command line argument}

# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def check_command_line_arguments(in_args):
    '''
      This function does some validation on the inputs provided via the command line.
        Parameters:    in_args   -  the input argument dictionary
        Returns:       None
    '''
    all_is_valid = True
    if (in_args['learn'] <= 0) or (in_args['learn'] > 1):
        print("You have entered an invalid learning rate.  Possible values are real numbers where (0 < value <= 1).")
        all_is_valid = False
    if (in_args['epochs'] < 1) or (in_args['epochs'] >= 50):
        print("You have entered an invalid no. epochs.  Possible values are integers where (1 < value <= 50).")        
        all_is_valid = False 
    if in_args['hidden_units'] < 1:
        print("You have entered an invalid no. of hidden units.  Possible values are integers where (1 <= value).")        
        all_is_valid = False       
    if not in_args['arch'] in ('resnet50', 'resnet152', 'densenet121', 'alexnet'):
        print("You have entered an unsuitable architecture for this application.")
        print("Suitable values are ('resnet50', 'resnet152', 'densenet121', 'alexnet').")        
        all_is_valid = False  
    
        
    if in_args['gpu'] == True:           # user wants GPU on for this job
        if not torch.cuda.is_available():
            print("You wanted Cuda turned on - however, it is currently off!!")
            print("Please turn it on before re-running this request!")
            print("Alternatively, run the job on the cpu - but then do not use the flag --gpu   Taking this option - expect significantly slower performance.")
            print("Exiting now...")
            exit()
    if not all_is_valid:
        exit()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #         
def create_datasets(in_args):
    """
    This function creates three datasets - one each for training data, validation data and testing data
    It then loads them into data loaders. 
      Parameters:    in_args   -  the input argument dictionary
      Returns:       The three data loaders: trainloader, validationloader & testloader.   Also, a mapping of classes to indices (taking the test_set)
   
    """
    data_dir = in_args['data_dir']
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # DEFINE TRANSFORMS & LOAD IMAGES INTO DATASETS - USING THESE TRANSFORMS
    ### TRAINING DATA....
    train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0), # Randomly change brightness/ contrast / saturation of an image.
    transforms.RandomHorizontalFlip(p=0.3),                                # Horiz. flip with prob. indicated
    transforms.RandomVerticalFlip(p=0.3),                                  # Vert. flip with prob. indicated
    transforms.RandomRotation((0, 360)),                                   # Rotate the image by angle between 0 & 360 degrees.
    transforms.RandomResizedCrop(224),                                     # Random crop down to size = (224*224)
    transforms.ToTensor(),                                                 # transform to a tensor...
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                     
                         std=[0.229, 0.224, 0.225])                     
    ])                    # ...tfm tensor with means & std devns to normalise each channel of the input

    train_set = datasets.ImageFolder(train_dir, transform=train_transforms) 
    # gets data from training dir to be processed with the transforms 'train_transforms'
    # (images are auto-labelled with the name of their directory)

    ### VALIDATION DATA....
    valid_transforms = transforms.Compose([
    transforms.Resize(255),                                                # Resize to  (255*255)
    transforms.CenterCrop(224),                                            # Centre crop to size = (224*224)
    transforms.ToTensor(),                                                 # transform to a tensor..
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                    
                         std=[0.229, 0.224, 0.225])                      
    ])
       
    valid_set = datasets.ImageFolder(valid_dir, transform=valid_transforms) 
    # gets data from validation dir to be processed with the transforms 'valid_transforms'
    
    ### TESTING DATA....
    test_transforms = transforms.Compose([
    transforms.Resize(255),                                                # Resize to  (255*255)
    transforms.CenterCrop(224),                                            # Centre crop to size = (224*224)
    transforms.ToTensor(),                                                 # transform to a tensor...
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                       
                         std=[0.229, 0.224, 0.225])                       
    ])
    
    test_set = datasets.ImageFolder(test_dir, transform=test_transforms) 
    # gets data from test dir to be processed with the transforms 'test_transforms'   

    
    # LOAD EACH DATASETS INTO ITS OWN DATALOADER using ImageFolder
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=46, shuffle=True)      # randomly shuffles for every epoch   
    validationloader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True) # randomly shuffles for every epoch
    testloader = torch.utils.data.DataLoader(test_set, batch_size=6, shuffle=True)         # randomly shuffles for every epoch
    
    return trainloader, validationloader, testloader, test_set

# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def get_cat_to_name_dict():
    """
    This function gets the file cat_to_name.json in the current directory & loads it into a dictionary. 
    This dict maps the integer encoded categories to the actual names of the flowers.
      Parameters:    None 
      Returns:       None
    """
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)              # reading the dict info into cat_to_name 
        
    return cat_to_name

# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def load_checkpoint(filename, in_args):
    '''
    This function takes the filename/path of a checkpoint & loads 
    - the MODEL that created the checkpoint, together with
    - the ASSOCIATED (WEIGHTS & BIASES) from the model's state_dict
      Parameters:    The checkpoint filename and in_args 
      Returns:       The model & optimizer and also checkpoint info (created_from_ch_pt, creating_arch, hidden_units)
    ''' 
    print("\nLoading your checkpoint model...")
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
                      
    checkpoint = torch.load(filename, map_location=map_location)  # 1 - load 'checkpoint' dict from file, into memory
                                                                
    
    my_learn_rate = in_args['learn'] 
    criterion = checkpoint['criterion'] 
    
    print("This checkpoint was created using ",checkpoint['architecture'])
    print("Consequently, the checkpoint will be loaded, making use of ",checkpoint['architecture'])
    print("This system will not allow you to amend the architecture or hidden layers as you are starting with a checkpoint - where structure & W&B dict must equate.")
    print("You may, however, select --epoch --learn --gpu --save_dir")
    print("If you wish to build a model using another architecture, then do not use the --chk argument")
    print("Best practice is to label your checkpoints that you create with the name of the architecture that was used in creating them.")
    hidden_units = checkpoint['fc1']
    model = getattr(torchvision.models, checkpoint['architecture'])(pretrained=True) # KEY!! 
   
    if checkpoint['architecture'] == 'alexnet' or checkpoint['architecture'] == 'densenet121':
        model.classifier = checkpoint['classifier']                                 # replace existing 'classifier' denoted "classifier" in model defn, with mine
        optimizer = optim.SGD(model.classifier.parameters(), lr=in_args['learn'])   # caters for other arch's with a 'classifier' 
    else:        
        model.fc = checkpoint['classifier']                                         # replace existing 'classifier' denoted "fc" in model defn, with mine
        optimizer = optim.SGD(model.fc.parameters(), lr=in_args['learn'])           # caters for resnet arch with an 'fc'
    
    model.load_state_dict(checkpoint['state_dict'])          # 3 - load the state_dict from the checkpoint into the model's state_dict
    model.class_to_idx = checkpoint['class_to_idx']          # 3 - load mapping of classes to index from checkpoint into the model
    model.epochs = in_args['learn']                          # 4 - load no epochs requested into model
    
    optimizer.load_state_dict(checkpoint['optimizer'])       # 4 - load optimizer's state_dict into memory
    
    print("\nYour checkpoint has now been loaded.")
    created_from_ch_pt = True
    creating_arch = checkpoint['architecture']
    answer = input("Do you wish to see the details of your loaded model?: (Y or N)") 
    if answer == 'Y' or answer == 'y':
        print("Your model is :\n")
        print(model)
        answer = input("\n\nDo you wish continue?: (Y or N)") 
        if answer != 'Y' and answer != 'y':
            print("Exiting now...")
            exit()
    else: 
        print("No problem...process continuing...\n\n")
        
    return model, optimizer, created_from_ch_pt, creating_arch, hidden_units

# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def train_model(in_args, model, optimizer, trainloader, validationloader, created_from_ch_pt, creating_arch):
    '''
     This function trains the loaded model using the hyper-parameters input in the command line.
    - the MODEL that created the checkpoint, together with
    - the ASSOCIATED (WEIGHTS & BIASES) from the model's state_dict
      Parameters:    the input arguments, model, optimizer, the trainloader and the validationloader
      Returns:       The trtained model & optimizer, and also checkpoint info (created_from_ch_pt, creating_arch)
    ''' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU access T-eed up
    print("Device in use is : ", device)
                  
    # select loss type; # select optimiser & learn rate
    criterion = nn.NLLLoss()  
    my_learn_rate = in_args['learn']               # Extracting argument from in_args - it is either the default or the entered value
     
    ##  Training the model  ##
    train_losses = []
    val_losses = []                                
    epochs = in_args['epochs']                     # Extracting argument from in_args - it is either the default or the entered value 
    
    model.to(device)                                        # move the processing to the GPU (if avail)
    start = time()                                          # start a Training Loop timer
    if created_from_ch_pt:                                  
        print("TRAINING: Architecture & hidden units being used, is that which was used in the creation of the checkpoint : i.e. ", creating_arch)
    else:
        print("TRAINING: Architecture '--arch' being used : ", in_args['arch'])
        print("TRAINING: Hidden units '-- hidden_units' being used : ", in_args['hidden_units'])
    print("TRAINING: No. of epochs '--epochs' being used : ", epochs)
    print("TRAINING: Learning rate '--learn' being used : ", my_learn_rate)
    print("\nNow Training....")
    for e in range(epochs):
        running_loss = 0
        no_batches = 0
        print("Starting Epoch no: {}".format(e+1))
        
        model.train()
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
                print("Training loss after {:3d} batches in epoch {:2d}, is {:.3f}".format(no_batches, e+1, running_loss/no_batches))
            # at this stage ALL IMAGES (all batches) IN TRAINLOADER have been run through...
                
        else:       #  If no bk command in the for loop, then the else part will be called.
            print("Training loss after {:2d} epochs is {:.3f}\n".format(e+1, running_loss/len(trainloader)) )
        
            val_loss = 0
            accuracy = 0
            no_batches = 0
            print("Now validating after epoch {:2d}".format(e+1))
        
            ##  Doing validation - using validation data after each epoch  ##
            model.eval()
            with torch.no_grad():                            # turn off gradients for validation for speed/memory
                for images, labels in validationloader:      # for all 13 batches in the validation data
                    images, labels = images.to(device), labels.to(device)   # ensure images & labels are also moved to GPU
                    ### Validation pass  ###  
                    logps = model(images)                   # FORWARD PASS to get log probs (logsoftmax used)
        
                    no_batches += 1
                    val_loss += criterion(logps, labels)    # calc & accumulate losses 
                    ps = torch.exp(logps)                   # get the probabilities back from the log probs
        
                    top_k, top_class = ps.topk(1, dim=1)    # get the predicted class no. of the highest prob into top_class
                    # create a 'byte tensor' called equals where (top_class == labels)  {labels had shape (64) resized to be same as top_class shape}
                    equals = top_class == labels.view(*top_class.shape)     # 'True' if predicted class matches true class
                    accuracy += torch.mean(equals.type(torch.FloatTensor))  # mean does not work on a byte tensor...make equals a float tensor
                    # torch.mean() returns a scalar tensor, to get the actual value as a float we'll need to do accuracy.item().
                    print("Batch No.: {:02d}".format(no_batches), "  Test/validation Loss: {:.3f}".format(val_loss/no_batches),"  Rolling Validation Accuracy {:.2f}%".format(accuracy.item()*100/no_batches))
        
            train_losses.append(running_loss/len(trainloader))
            val_losses.append(val_loss/len(validationloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f} ".format(running_loss/len(trainloader)),
                  "Test/validation Loss: {:.3f} ".format(val_loss/len(validationloader)),
                  "Test/validation Accuracy: {:.2f}%".format(accuracy.item()*100/len(validationloader)))
            #print(train_losses)     # for testing
            #print(val_losses)       # for testing
   
    stop = time()                                      # stop the Training/Validation Loop timer 
    print("Model training has now completed.\n\n")
    print("FURTHER STATISTICS FROM THE TRAINING")
    print("Total time training & validating (in seconds): {:.3f}".format((stop - start)))
    time_per_epoch = (stop - start)/epochs
    print("Device is: {};  Time per epoch: {:.3f} seconds".format(device, time_per_epoch))  # print ave time/epoch 
    time_per_batch = time_per_epoch/no_batches
    print("Ave time per batch : {:.3f} seconds.".format(time_per_batch))
    print("\n\n")
    
# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def save_checkpoint(in_args, test_set, model, optimizer, created_from_ch_pt, creating_arch, hidden_units):
    '''
    This function saves a checkpoint.  It is typically used after training has completed.
      Parameters:    the input arguments, model, optimizer, the test_set, and also checkpoint info (created_from_ch_pt, creating_arch, hidden_units)
      Returns:       None
    '''
    print("Your trained model checkpoint is being saved now...")
    
    in_features = {"vgg16":25088, "densenet121":1024, "alexnet":9216, "resnet50":2048, "resnet152":2048}
    
    
    # if created from a checkpoint........then use orig model's arch & hu's - not those from the command line
    if created_from_ch_pt:
        if creating_arch == 'resnet50' or creating_arch == 'resnet152':   
            model.class_to_idx = test_set.class_to_idx   # a mapping of classes to indices   (taking the test_set)
            checkpoint = {'input_size' : in_features[creating_arch],
                          'fc1' : hidden_units,
                          'fc2' : 512,
                          'fc3' : 102,
                          'output_size' : 102,
                          'architecture' : creating_arch,                  # saving this means that you dont need to re-load the model from torchvision models later
                          'state_dict' : model.state_dict(),
                          'optimizer' : optimizer.state_dict(),
                          'classifier' : model.fc,                           # res* uses fc
                          'epochs' : in_args['arch'],
                          'my_learn_rate' : in_args['learn'],
                          'criterion' : 'nn.NLLLoss()',
                          'class_to_idx': model.class_to_idx                 # mapping of classes to index
                         }
        else:  
            model.class_to_idx = test_set.class_to_idx                       # a mapping of classes to indices   (taking the test_set)
            checkpoint = {'input_size' : in_features[creating_arch],
                          'fc1' : hidden_units,
                          'fc2' : 512,
                          'fc3' : 102,
                          'output_size' : 102,
                          'architecture' : creating_arch,                  # saving this means that you dont need to re-load the model from torchvision models later
                          'state_dict' : model.state_dict(),
                          'optimizer' : optimizer.state_dict(),
                          'classifier' : model.classifier,                   # other archtectures use classifier
                          'epochs' : in_args['arch'],
                          'my_learn_rate' : in_args['learn'],
                          'criterion' : 'nn.NLLLoss()',
                          'class_to_idx': model.class_to_idx                 # mapping of classes to index
                         }    
        
    
    else:      # if built from scratch........then use the command line --arch & --hidden_units
        if in_args['arch'][0:3] == 'res':   
            model.class_to_idx = test_set.class_to_idx   # a mapping of classes to indices   (taking the test_set)
            checkpoint = {'input_size' : in_features[in_args['arch']],
                          'fc1' : in_args['hidden_units'],
                          'fc2' : 512,
                          'fc3' : 102,
                          'output_size' : 102,
                          'architecture' : in_args['arch'],                  # saving this means that you dont need to re-load the model from torchvision models later
                          'state_dict' : model.state_dict(),
                          'optimizer' : optimizer.state_dict(),
                          'classifier' : model.fc,                           # res* uses fc
                          'epochs' : in_args['arch'],
                          'my_learn_rate' : in_args['learn'],
                          'criterion' : 'nn.NLLLoss()',
                          'class_to_idx': model.class_to_idx                 # mapping of classes to index
                         }
        else:  
            model.class_to_idx = test_set.class_to_idx                       # a mapping of classes to indices   (taking the test_set)
            checkpoint = {'input_size' : in_features[in_args['arch']],
                          'fc1' : in_args['hidden_units'],
                          'fc2' : 512,
                          'fc3' : 102,
                          'output_size' : 102,
                          'architecture' : in_args['arch'],                  # saving this means that you dont need to re-load the model from torchvision models later
                          'state_dict' : model.state_dict(),
                          'optimizer' : optimizer.state_dict(),
                          'classifier' : model.classifier,                   # other archtectures use classifier
                          'epochs' : in_args['arch'],
                          'my_learn_rate' : in_args['learn'],
                          'criterion' : 'nn.NLLLoss()',
                          'class_to_idx': model.class_to_idx                 # mapping of classes to index
                         }      
        
    save_dir = in_args['save_dir']                       # Extract the directory path to which user wishes to save
    torch.save(checkpoint, save_dir)                     # Save to that chosen directory
    print("...being saved as {}".format(save_dir)) 
   
# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def build_model(in_args):
    '''
    This function gets the requested pretrained model from torchvision.models (in order to get the image features). 
    It build a new feed-forward classifier using those features from the pretrained model & using the specified no. of hidden units.
    It selects the optimizer SGD.
      Parameters:    the input arguments 
      Returns:       the built model and selected optimizer
    '''
    in_features = {"vgg16":25088, "densenet121":1024, "alexnet":9216, "resnet50":2048, "resnet152":2048}
    
    # LOAD A PRE-TRAINED MODEL           # has a 'features' component & a 'classifier' component
    if in_args['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif in_args['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif in_args['arch'] == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif in_args['arch'] == 'resnet50':
        model = models.resnet50(pretrained = True)
    elif in_args['arch'] == 'resnet152':
        model = models.resnet152(pretrained = True)
        
    # FREEZE the 'FEATURES' PARAMETERS (because we don't want these to get updated) & this speeds up training
    for param in model.parameters():
        param.requires_grad = False                     # turn off gradient calculation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU access T-eed up
    print("\nDevice used to build the model is : ", device)


    # CREATE MY CLASSIFIER  (we want to REPLACE their 'classifier' with our own one ....)       
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([            # naming my transforms & operations with OrderedDict()
        ('fc1', nn.Linear(in_features[in_args['arch']], in_args['hidden_units'])),     # my model design of fully-connected layers
        ('relu1', nn.ReLU()),                              # starting with orig classifier's 'in_features=25088'?????too slow
        ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(in_args['hidden_units'], 512)),  # my model design of fully-connected layers
        ('relu2', nn.ReLU()),                              # starting with orig classifier's 'in_features=25088'?????too slow
        ('dropout2', nn.Dropout(p=0.1)),
        ('fc3', nn.Linear(512, 102)),                      # 102 = no. of possible flower names
        ('output', nn.LogSoftmax(dim=1)),]))
    
    if in_args['arch'][0:3] == 'res':
        model.fc = classifier                                                      # replace existing 'classifier' denoted "fc" in model defn, with mine
        optimizer = optim.SGD(model.fc.parameters(), lr=in_args['learn'])          # use SGD optimizer & get the specified learn rate 
    else:
        model.classifier = classifier                                              # replace existing 'classifier' denoted "classifier" in model defn, with mine
        optimizer = optim.SGD(model.classifier.parameters(), lr=in_args['learn'])  # use SGD optimizer & get the specified learn rate                
    print("model first three chars is ", in_args['arch'][0:3])
    print("No. of hidden units is :", in_args['hidden_units'])
    print("Architecture is :", in_args['arch'])
    
      
    answer = input("Would you like to see the model that has been built? (Y or N)")
    if answer == "Y" or answer == "y":
        print(model)
    answer = input("Would you like to continue? (Y or N)")
    if answer == "N" or answer == "n":
        print("Exiting now...")
        exit()   
    
    return model, optimizer    