# PROGRAMMER:   Michael Gaylard
# DATE CREATED: 8th October 2020                                 
# REVISED DATE: 12th October 2020
# PURPOSE: train.py has a 3-pronged purpose: 
#             1. Train a new network on a dataset;
#             2. Print out training loss, validation loss, and validation accuracy as the network is being trained.
#             3. Save the model as a checkpoint after training.

#          Inputs:  
#            This script is run from the command line - with the following input arguments:  (<> indicates expected user input)
#          Basic Usage:    
#            python train.py <data_directory>                                                  
#          Further options:
#            python train.py <data_directory> --save_dir <path for saving checkpoints>  
#            python train.py <data_directory> --arch <model>
#            python train.py <data_directory> --learning_rate <learning rate required>
#            python train.py <data_directory> --hidden_units <no. hidden units required>
#            python train.py <data_directory> --epochs <no. epochs required>
#            python train.py <data_directory> --gpu
#          Example call from the command line:
#            python train.py flowers --save_dir checkpoint9.pth --arch vgg --learning_rate 0.4 --hidden_units 512 --epochs 16 --gpu
#   
##
# Import python modules
from time import time

# Import supporting functions created for this program
from t_setup import get_input_args, check_command_line_arguments, create_datasets, get_cat_to_name_dict, load_checkpoint, train_model, save_checkpoint, build_model

# Main program function defined below
def main():
   
    # gets Command Line Arguments from the user.    
    in_args = get_input_args()       # get my input arguments
    print("These are the input arguments...")                                                                       
    print(in_args)                                                                                               

    # checks the validity of the command line arguments.  
    check_command_line_arguments(in_args)
    
    # accesses flower images from the data_sets path, transforms them & then loads them into dataloaders - ready for training.
    trainloader, validationloader, testloader, test_set = create_datasets(in_args) 

    # gets a dict mapping of the integer encoded categories to the actual names of the flowers
    cat_to_name = get_cat_to_name_dict()  
        
    # Additional feature --chk: prompts for, gets and loads a checkpoint file from a location as specified by the user 
    created_from_ch_pt = False
    creating_arch = ''
    hidden_units = 0
  
    if in_args['chk']:
        answer = input("Enter a valid checkpoint.pth filename which is in the current directory:  ")
        print("Note that the checkpoint will be loaded in the architecture in which it was created & no hidden units will be applied for training.")
        print("In other words - your --arch & --hidden_units will be ignored when training from a checkpoint.")
        model, optimizer, created_from_ch_pt, creating_arch, hidden_units = load_checkpoint(answer, in_args)
    else:
        model, optimizer = build_model(in_args)

    # Collects start time at start of training
    start_time = time()  
    
    # trains the model, and outputs training statistics (incl. training loss, validation loss and validation accuracy)
    # Validation is done after each epoch, making use of validation data
    train_model(in_args, model, optimizer, trainloader, validationloader, created_from_ch_pt, creating_arch)

    # saves the final checkpoint after training completion - into the location as specified by the user as one of the input arguments
    # If no such argument is found, then the default loacation checkpoint99.pth is used
    save_checkpoint(in_args, test_set, model, optimizer, created_from_ch_pt, creating_arch, hidden_units)    

    # Collects end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format $$
    tot_time = end_time - start_time  #  program run-time is calculated in seconds
    # Converts run-time to hh:mm:ss format and prints it
    print("\n** Total Elapsed Runtime : [hh:mm:ss]",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
#______________________________________________________________________________________
