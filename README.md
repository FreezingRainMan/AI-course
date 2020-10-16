# AI-course
Udacity AI course
Project to build a command line initiated classifier.
TRAIN:   train.py builds a network (or allows you to grab an alrady built checkpoint) loads it and trains it according to command line arguments.
All the functions that it uses, are defined in & drawn from t_setup.py

The basic command is $ python train.py <data_dir> 
The accepted command line arguments are: 
--epoch <#>             being no. epochs (1-20) to use in training
                        Default: 2
--learn <#>             being the learn rate (0 to 1) to be used in training
                        Default: 0.15
--arch <str>            being the name of the architecture you are wanting to use   (only a few valid values)
                        Default: resnet50
--hidden_units <#>      being the number of hidden units to use in a model that you build
                        Default: 1012
--save_dir <str>        being the path where you wish the post-training checkpoint to be saved.    Must have extn .pth
                        Default: checkpoint100.pth
--gpu                   being a flag indicating that MUST have the gpu running for this job 
                        Default: set to not required
The model is built & loaded.
Image data is grabbed and processed and then loaded into dataloaders (test, train, validate).
Then the training is done.  During training, Training Loss is shown after every 10 batches.  
After every epoch, the model is tested against the validation data.  The accuracy against the validation data is shown for every 10 batches tested.
After training the checkpoint of the trained model is saved.

There is also the possibility of loading an existing checkpoint and continuing the training using this.
To initiate this option, use the --chk option.  It will prompt you for the checkpoint you wish to load.  If it is a valid checkpoiont, it will be loaded.
When loading a checkpoint, the architecture & hidden layers of that checkpoint are used and any --arch or --hidden_units you may have entered will be ignored.
This is because the model and the Weights & Biases must be aligned.  The other arguments will be used as entered.

PREDICT: predict.py takes in an image & a model.  It runs the image through the model & predicts the flower in the image.
All the functions that it uses, are defined in & drawn from p_setup.py

The basic command is $ python predict.py <image path>  <checkpoint path>
The accepted command line arguments are: 
--image_dir <str>         being the directory name in the current folder that houses all the images (such images to have subfolders mirroring those in the flowers folder)
                          Default: randomly selected from flowers/test/
--chk <str>               being the filename of the checkpoint that should be used to load for prediction 
                          Default: ch100-resnet50.pth
--top_k <#>               being the no. of top probabilities that you wish to see for this image 
                          Default: 5
--gpu                     being a flag indicating that MUST have the gpu running for this job 
                          Default: set to not required
--category_names <str>    being the path to the file that houses the cat_to_name.json file (has the category : label name information)
                          Default: ./cat_to_name.json
