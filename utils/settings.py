#I stored some parameters that get commonly passed to functions here, so you can simply pass the settings to functions
#You can change the defaults of the parameters, especially the path ones

class settings():
    def __init__(self,
                classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'],
                dataset_path = '/home/brian/Data/APS360/APS Project/PlantVillage',
                features_path = '/home/brian/Data/APS360/APS Project/PlantVillage_Features',
                tensorboard_logdir = "/home/brian/Data/APS360/APS Project/logs",
                use_cuda=False,
                learning_rate = 1e-3,
                num_epochs = 30,
                save_weights = False):
        self.classes = classes
        self.dataset_path = dataset_path
        self.features_path = features_path #this one is optional if not doing feature extraction
        self.tensorboard_logdir = tensorboard_logdir
        self.use_cuda = use_cuda
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_weights = save_weights
    def num_classes(self): #get number of classes, used in some functions
        return len(classes)
    def show_settings(self): #print out current settings
        temp = vars(self)
        for item in temp:
            print(item,':',temp[item])
        return True