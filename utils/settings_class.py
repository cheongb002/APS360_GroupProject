#I stored some parameters that get commonly passed to functions here, so you can simply pass the settings to functions
#You can change the defaults of the parameters, especially the path ones
import os
from datetime import datetime
from pathlib import Path

class settings():
    def __init__(self,
                classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'],
                dataset_path = '/home/brian/Data/APS360/APS_Project/PlantVillage',
                features_path = '/home/brian/Data/APS360/APS_Project/PlantVillage_Features/efficientnet-b0',
                tensorboard_logdir = "/home/brian/Data/APS360/APS_Project/logs",
                weight_checkpoints = "/home/brian/Data/APS360/APS_Project/checkpoints",
                use_cuda=False,
                learning_rate = 1e-3,
                num_epochs = 30,
                batch_size = 16,
                save_weights = False,
                image_size = [224,224],
                save_freq = 1,
                identifier = None,
                train_val_test_split = [0.8,0.1,0.1],
                settings_path = "/home/brian/Data/APS360/APS_Project/trial_settings",
                randomRotate = False,
                randomHoriFlip = False,
                randomVertFlip = False,
                randomGray = False,
                randomCrop = False,
                randomBlur = False

                #add some boolean parameters for any transformations we want
                ):
        """Aggregated settings class for various scripts
        Args:
            classes (list): list of classes, can be obtained from dataset.classes
            dataset_path (string): Location of PlantVillage dataset
            features_path (string): Location of featuers generated from PV dataset
            tensorboard_logdir (string): location of where all the tensorboard files will be saved to during training
            weight_checkpoints (string): Where to save the weights if save_weights is True
            use_cuda (bool): Whether you want to use CUDA
            learning_rate (float): Learning rate for trainer
            num_epochs (int): Number of epochs to train for
            batch_size (int): Batch size for DataLoaders
            save_weights (bool): whether to save weights during training
            image_size (list of ints or int): Dimension(s) to resize images to. Must be specific to the desired model to train on
            save_freq (int): how often do you want to save the weights
            identifier (str): some string to describe the trial being run
            train_val_test_split (list of floats): ratio of each distribution desired
        """
        self.classes = classes
        self.dataset_path = dataset_path #where the datset is stored
        self.features_path = features_path #this one is optional if not doing feature extraction
        Path(self.features_path).mkdir(parents=True, exist_ok=True)
        self.tensorboard_logdir = tensorboard_logdir
        Path(self.tensorboard_logdir).mkdir(parents=True, exist_ok=True)
        self.weight_checkpoints = weight_checkpoints
        Path(self.weight_checkpoints).mkdir(parents=True, exist_ok=True)
        self.use_cuda = use_cuda
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_weights = save_weights
        self.image_size = image_size
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.identifier = identifier
        self.train_val_test_split = train_val_test_split
        self.randomRotate = randomRotate,
        self.randomHoriFlip = randomHoriFlip,
        self.randomVertFlip = randomVertFlip,
        self.randomGray = randomGray,
        self.randomCrop = randomCrop,
        self.randomBlur = randomBlur

        self.settings_path = settings_path
        Path(self.settings_path).mkdir(parents=True, exist_ok=True)

    def num_classes(self): #get number of classes, used in some functions
        #this is a function instead of a variable just in case you change classes
        return len(self.classes)
    
    def show_settings(self): #print out current settings
        temp = vars(self)
        for item in temp:
            print(item,':',temp[item])
        return True

    def save_settings(self): #save current settings to a txt file
        with open(os.path.join(self.settings_path, self.identifier+".txt"), "a") as text_file:
            print("\n Trial run at {}\n".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")), file=text_file)
            temp = vars(self)
            for item in temp:
                print(item, ":", temp[item], file=text_file)
            