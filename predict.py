# PROGRAMMER:   Michael Gaylard
# DATE CREATED: 9th October 2020                                 
# REVISED DATE: 12th October 2020
# PURPOSE: predict.py predicts flower name from an image, together with the probability of that name.
#          
#          Inputs: 
#            This script is run from the command line - with the following input arguments:  (<> indicates expected user input)
#          Basic Usage: 
#            python predict.py <image path> <checkpoint path>
#          Further options:
#            python predict.py <image path> <checkpoint path> --top_k <x>                           
#            python predict.py <image path> <checkpoint path> --category_names cat_to_name.json           
#            python predict.py <image path> <checkpoint path> --gpu                               
#          Example call from the command line:
#            python predict.py './image_01.jpg' 'checkpoint99.pth' --top_k 3 --category_names cat_to_name.json  --gpu      
#         
##
# Import python modules
# none

# Import supporting functions created for this program
from p_setup import get_image, process_image, load_checkpoint, get_cat_to_name_dict, predict, get_input_args, check_command_line_arguments

# Main program function defined below
def main():
    
    # gets Command Line Arguments from the user.    
    in_args = get_input_args()       # get my input arguments
    print("These are the input arguments (&/or their defaults)")                                                                       
    print(in_args)                     # printed for visibility 
    print("\n")
          
     # checks the validity of the command line arguments. 
    check_command_line_arguments(in_args)
           
    # load_checkpoint() - gets and loads the checkpoint file from the location as specified by the user as one of the input arguments 
    # If no such argument is found, then the default data file (checkpoint7.pth) is used 
    model, optimizer = load_checkpoint(in_args)
    
    # gets and loads the image file from the location as specified by the user as one of the input arguments 
    # If no such argument is found, then a random image file is used 
    img_path = get_image(in_args)
    
    # loads in a mapping from the category to name rom the location as specified by the user as one of the input arguments.
    # If no such argument is found, then the file  ./cat_to_name.json is used 
    cat_to_name = get_cat_to_name_dict(in_args) 
    
    # determines & prints top k probabilities & their labels for the image.
    # determines & shows correct name for flower.
    top_5_probs, top_5_class_nums = predict(img_path, model, cat_to_name, in_args)      

# Call to main function to run the program
if __name__ == "__main__":
    main()
#______________________________________________________________________________________
