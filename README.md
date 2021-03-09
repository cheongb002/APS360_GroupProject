# APS360_GroupProject
Creating a machine learning model to classify images from the Plant Village 

## Summary of available folders:

- scripts: Files that can be called directly from a terminal or run to do something.
- utils: stores files containing functions that the various scripts will be using
- efficientnet_pytorch: Contains code from the https://github.com/lukemelas/EfficientNet-PyTorch.git repository for EfficientNet model.

Note to self: if encountering no module error when running a file in terminal, run export PYTHONPATH=. (period required)

### About scripts:

Contains the files that can be directly run in terminal or command prompt.

Please see the train_script_template.py for how to write new training scripts.

### About utils:

Contains the files with functions and classes that will be used by the scripts.

- common.py: random functions, like get_model_name, get_accuracy, create_model. NOTE: create model is where we should put the code to instantiate models. Scripts should be able to simply call create_model without any overhead.

- models.py: Where you can store your model classes, to be called by create_model

- settings.py: the settings class to hold all of the training settings. You can change the defaults to anything, especially the defaults of what folders things are stored in. Most functions will just take settings as a parameter and access the variables through the object.

- train_utils.py: contains the train_net function only for now, we can add training related functions later down the line.

## How to use Tensorboard
(idk if this works on colab, you might have to jump through some hoops)
If you want to be able to see graphs of your training in real time, cd into your log directory, and run `tensorboard --logdir=<trial identifier>` to create a link to a site that will have certain graphs. It's limited for now, but it isn't hard to track other values during training. Note, you will need to get the tensorflow and tensorboard packages, easily available through anaconda or pip.