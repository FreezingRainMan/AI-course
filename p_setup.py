# # PROGRAMMER: Michael Gaylard
# DATE CREATED: 9th October 2020                                 
# REVISED DATE: 
# PURPOSE: p_setup.py provides the following functions that are used by the script 'predict.py'        
#          These functions are:
#             get_input_args()                    : Retrieves Command Line Arguments from user & returns these (or the defaults), as the variable in_args                   #             check_command_line_arguments()      : Validates Command Line Arguments and exits if validation rules are broken
#             get_image()                         : Gets the image from the chosen file (or randomly chosen)
#             load_checkpoint()                   : Loads the checkpoint (Model & Optimizer) from the chosen file (or the in_args default)
#             process_image()                     : Processes the image for running through the model
#             get_cat_to_name_dict                : Loads the mapping (cat. no. to cat. name) of the test data into a dictionary  
#             predict()                           : Predicts the chosen k most probable classes (or the in_args default for k)

# Imports python modules
import argparse                                   # for getting inputs from the command line & subsequently parsing them
import os                                         # to perform validity check on file paths captured on the command - to check that they exist
import random                                     # to randomly get images
from time import time                             # to time the program
import torch
from torchvision import datasets, transforms      # to create datasets
import torchvision                                # required to re-create the architecture
from torch import nn, optim                       # for model classifier design & loss & optimiser choice
from PIL import Image                             # for processing images
import numpy as np                                # for processing images
import json                                       # to access the cat_to_name_dict info
from torch.autograd import Variable               # to do the fwd pass in the predict function

# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when they run the program from a terminal window. 
    This function uses Python's argparse module to created and define these command line arguments.
    If the user fails to provide some or all of the 6 arguments, then the default values are used for the missing arguments. 
    Command Line Arguments:
     1. Image path <image path>                                any valid image path & filename               default random image
     2. Checkpoint path --chk <checkpoint path>                any valid checkpoint path & filename          default "./ch100-resnet50.pth" 
     3. Top k probabilities --top_k < >                        possible values of (1 < value <= 20)          default 5
     4. Mapping of categories to names  --category_names < >   any valid path & filename ending in .json     default "cat_to_name.json" 
     5. GPU requested as --gpu                                 n/a                                           flag defaulted to 'gpu not required'
    This function returns these arguments as a dictionary.
        Parameters:    None - simply using argparse module to create & store command line arguments
        Returns:       in_args() - dictionary that stores the command line arguments names & values 
    """
    # Instantiates an ArgumentParser object called 'parser'
    parser = argparse.ArgumentParser()

    # Creates POSITIONAL command line arguments, using add_argument() method from ArgumentParser 
    parser.add_argument("image_dir",        help = "Path to get the image you wish to evaluate",    nargs = '?', type = str, default = 'get random')
    #nargs = '?'  means num of arguments could be one or none - in which case the default is used
    
    # Creates OPTIONAL command line arguments, using add_argument() method from ArgumentParser 
    parser.add_argument("--chk",            help = "Path to the model checkpoint you wish to load", nargs = '?', type = str, default = './ch100-resnet50.pth')
    parser.add_argument("--top_k",          help = "Top k class probabilities",                                type = int, default = 5)
    parser.add_argument("--gpu",            help = "Sets flag that gpu should be used",                        action='store_true') 
    parser.add_argument("--category_names", help = "Path to get a specific category names file",  nargs = '?', type = str, default = './cat_to_name.json')

    
    in_args = parser.parse_args() 
#    # Returns the parsed argument collection created within this function 
#    # this 'parsed argument collection' is a 'Namespace' of format Namespace(dir = 'mydir', arch = 'myarch', dogfile = 'mydogfile')
    in_args = vars(in_args)
#    # Converts the Namespace into a dictionary such as {'arch' : 'vgg13', 'learn' : '0.4', 'gpu' : 'gpu'} 
#    # for easier accessing later.  e.g. using in_args['dir'] to return 'mydir'
#   
    return in_args        #  in_args is a dict containing pairs of {name of command line argument : value of command line argument}
#
# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def check_command_line_arguments(in_args):
    '''
    This function does some validation on the inputs provided via the command line.
        Parameters:    in_args   -  the input argument dictionary
        Returns:       None
    '''
    all_is_valid = True
    if (in_args['top_k'] <= 0) or (in_args['top_k'] > 20):
        print("You have entered an invalid top k.  Possible values are real numbers where (0 < value <= 20).")
        all_is_valid = False   
 
    if not os.path.exists(in_args['image_dir']) and in_args['image_dir'] != 'get random':
        print("Image path specified - is Invalid...")
        all_is_valid = False
    
    if in_args['image_dir'] == 'get random':
        print("No image was specified.  If you continue, a random image will be automatically selected.")
        answer = input("Do you wish to continue with such a random image selection? (Y or N)")
        if answer == 'n' or answer == 'N':
            all_is_valid = False   
        
    if not os.path.exists(in_args['chk']):
        print("Checkpoint path specified - is Invalid...")
        all_is_valid = False
        
    if not os.path.exists(in_args['category_names']):
        print("Category_names path specified - is Invalid...")
        all_is_valid = False
           
    if in_args['gpu'] == True:           # user wants GPU on for this job
        if not torch.cuda.is_available():
            print("You wanted Cuda turned on - however, it is currently off!!")
            print("If you really wish it to be on - then please turn it on when re-running this request with the argument --gpu !")
            print("Alternatively, let the job continue on the cpu.")
            answer = input("Do you wish to continue - just using cpu? (Y or N):")
            if answer == 'N' or answer == 'n':     
                all_is_valid = False
    if not all_is_valid:
        print("The general format is 'python predict.py image_path checkpoint path' & options are --top_k and --gpu")
        print("Exiting now...")
        exit()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def get_image(in_args):
    ''' 
    Get the image as requested, or randomly get an image if no specific image was chosen in the command line.
      Parameters:    in_args   -  the input argument dictionary
      Returns:       a path to the image
    '''
    if in_args['image_dir'] == 'get random':                              # if no image path was captured in the command line - get a random image...   
        # Get random image from the test images folders
        print("Getting a random image, as nothing was specified...")
        rd_folder = random.randint(1,99)
        img_path_so_far = './flowers/test/' + str(rd_folder) + "/"     # randomise the numbered subfolder of test/
        img = random.choice(os.listdir(img_path_so_far))               # randomise the file in folder test/random/
        img_path = img_path_so_far + img
    else:
        print("Getting the specified image...     ", end='')
        print(in_args['image_dir'])
        img_path =in_args['image_dir']                                   # ...else use the image path as specified in the command line
    
    return img_path
    
# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #        
def load_checkpoint(in_args):
    '''
    Takes the filename/path of a checkpoint & loads 
    - the MODEL that created the checkpoint, together with
    - the ASSOCIATED (WEIGHTS & BIASES) from the model's state_dict
    If no checkpoint is specified in the command line, then a default checkpoint is offered.
    ''' 
    print("Loading your checkpoint model...")
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    if in_args['chk'] == './ch100-resnet50.pth':      # if the defaulted checkpoint is used
        print("...No checkpoint model was specified.")
        print("The default checkpoint is  './ch100-resnet50.pth'")     
        answer = input("Do you wish to continue with this default?: (Y or N)") 
        if answer == 'N' or answer == 'n':
            print("Exiting now...")
            exit()
   
    checkpoint = torch.load(in_args['chk'], map_location=map_location)  # Access the checkpoint from the command line arguments
                                                                
    my_learn_rate = checkpoint['my_learn_rate'] 
    criterion = checkpoint['criterion'] 
    
    model = getattr(torchvision.models, checkpoint['architecture'])(pretrained=True) # KEY!! 
    
    if checkpoint['architecture'] == 'alexnet' or checkpoint['architecture'] == 'densenet121':
        model.classifier = checkpoint['classifier']                                         # replace existing 'classifier' denoted "fc" in model defn, with mine
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.1)              # arbitrary lr ... caters for resnet arch with an 'fc'
        optimizer.load_state_dict(checkpoint['optimizer'])                    # 4 - load optimizer's state_dict into memory# these 2 lines instead of re-creation !!!         
    else:                                                                    # I cater for only resnet50 & resnet152 apart from the 'others'
        model.fc = checkpoint['classifier']  
        optimizer = optim.SGD(model.fc.parameters(), lr=0.1)                  # arbitrary lr ... caters for non resnet arch with a 'classifier'
        optimizer.load_state_dict(checkpoint['optimizer'])                    # 4 - load optimizer's state_dict into memory# these 2 lines instead of re-creation !!!
        
    model.load_state_dict(checkpoint['state_dict'])          # 3 - load the state_dict into the model's state_dict
    model.class_to_idx = checkpoint['class_to_idx']          # 3 - load mapping of classes to index into the model
    model.epochs = checkpoint['epochs']                      # 4 - load no epochs into model
       
    print("... {} has now been loaded.\n".format(in_args['chk']))
   
    return model, optimizer

# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model, & returns an Numpy array
      Parameters:    a path to the image
      Returns:       a processed image as a numpy array
    '''
    image = Image.open(image)                      # convert the str into resizeable & croppable thing
    image = image.resize((256,256)) 
    cr = 0.5*(256-224)
    image = image.crop((cr, cr, 256-cr, 256-cr))
    
    image = np.array(image)                       # convert into an ndarray for normalisation
    image = image/255. 
    mean = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])
    image = ((image - mean) / sd)
    
    image = np.transpose(image, (2, 0, 1))       # transpose to meet the demands on matplotlib channel position
    return image                                 # an image as an ndarray
# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def get_cat_to_name_dict(in_args):
    """
    Loads in a mapping from cat. no. to cat. name.  Held in file cat_to_name.json in the current directory
    This provides a dict mapping the integer encoded categories to the actual names of the flowers
      Parameters:    in_args   -  the input argument dictionary
      Returns:       a category to name dictionary
    """
    with open(in_args['category_names'], 'r') as f:
        cat_to_name = json.load(f)              # reading the dict info into cat_to_name  
            
    return cat_to_name

# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def predict(image_path, model, cat_to_name, in_args, topk=5):
    ''' 
    Predicts the top n classes of an image, from an image file, using a trained deep learning model.
      Parameters:    the processed image, the model, the cat to name dictionary, in_args, topk
      Returns:       topk probabilities and class numbers
    '''  
     # move the model to cuda if possible
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()                              # if cuda then move model parameters to the GPU
        print("We are using GPU...")
    else:
        model.cpu()                               # else move move model parameters to the CPU
        print("We are using CPU...")

    # Process the image
    print("Processing the image for conformance with the model...")
    image = process_image(image_path)                 # processs the image & get back an ndarray
    # transfer ndarray to tensor before FWD RUN
    image = torch.from_numpy(np.array([image])).float()
    # The image becomes the input
    image = Variable(image)                           # still not clear on .Variable()
    if cuda:
        image = image.cuda()
    
    print("Evaluating the image ...\n")    
    model.eval()
    output = model.forward(image)                     # RUN FORWARD 
    probabilities = torch.exp(output).data            # 102 as expected
    
    if "top_k" in in_args:                            # Extract the no of top probabilities that may have been specified in the command line
        topk = in_args["top_k"]
       
    # getting the topk probabilities and indexes
    # 0 -> probabilities
    # 1 -> index used to look up class no. in model.class_to_idx       {topk returns a [[probs] [classes]] tensor}
    prob = torch.topk(probabilities, topk)[0].tolist()[0]               # get top k probs into a list & peel once    
    index = torch.topk(probabilities, topk)[1].tolist()[0]              # a real index is the model output num that is needed to lookup class
                                                           
    all_classes = []                                                  
    for i in range(len(model.class_to_idx.items())):                    # create a list of all the keys in model.class_to_idx.items()
        all_classes.append(list(model.class_to_idx.items())[i][0])      # call it all_classes
      
    class_nums = []
    for i in range(topk):
        class_nums.append(all_classes[index[i]])
                
    print("The top {} probabilities (to 4dp's) are:".format(topk))
    for p in range(len(prob)):
        print("{:02d} : {:.4f}".format(p+1, prob[p]))                                                          # print the list of the top k probabilities
    print("\nThe top {} predicted classes are:".format(topk))
    print(class_nums)                                                    # print the list of the top k class numbers
    
    print("\nThe names for the top {} predicted classes are:".format(topk))
    print([cat_to_name[x] for x in class_nums])                          # print class names associated to the top k probabilities
    print("\nTOP PREDICTION is:", cat_to_name[class_nums[0]].upper(), "\n")        # print class name of top probability  
      
    # Get & print correct flower class_num & name
    act_class_num = image_path.split('/')[-2]                            # returns folder no.
    print("CORRECT class no. is :", act_class_num.upper())
    print("CORRECT flower name :", cat_to_name[act_class_num].upper(), "\n")
          
    return prob, class_nums
# -------------------------------------------------------------------------------------------------------------------------------------------------------------- #



        