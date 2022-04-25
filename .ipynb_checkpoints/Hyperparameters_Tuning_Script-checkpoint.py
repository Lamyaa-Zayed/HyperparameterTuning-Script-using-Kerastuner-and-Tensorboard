'''
    This script is to build a general hyper parameter tuning optimization 
    method for tensorflow classification/regression models in order to 
    decrease the number of trainings. 
'''
#_________________________________________________________________________________
#
# Import packages
#
import sys, getopt
import tensorboard
import numpy as np
import pandas as pd
from datetime import datetime
#from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#
# Import Keras modules
#
import tensorflow as tf
import keras_tuner as kt
#from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input
#from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredLogarithmicError
#_________________________________________________________________________________

#
# Hyper Parameter Optimization Class
#
class HPO():
    
    # init function which takes any dataset and the hyperparameters arguments from user to be tuned
    def __init__(self, dataset, epochs, hp_tuner_type, hp_layers, 
                 hp_activation, hp_units, hp_dropout_rate, hp_learning_rate):
        
        # read csv file of dataset
        df = pd.read_csv(dataset)
        self.X = df.iloc[: , :-1]
        self.y = df.iloc[: , -1]
        
        # arguments default values
        self.max_trials = 10
        self.epochs = epochs
        self.hp_tuner_type = hp_tuner_type
        self.hp_layers = hp_layers
        self.hp_activation = hp_activation
        self.hp_dropout_rate = hp_dropout_rate
        self.hp_units = hp_units
        self.hp_learning_rate = hp_learning_rate
        
    # Create LogDir function to get all tuner and tensorboard logs
    def get_logdir(self):
        # Save the time for every log
        now = datetime.now()
        self.logdir_tuner = "logs/" + now.strftime("%Y%m%d-%H%M%S") + "/" + "Tuner"
        self.logdir_tensorboard = "logs/" + now.strftime("%Y%m%d-%H%M%S") + "/" + "TensorBoard"
     
    # Function to select Keras_Tuner types which are (RandomSearch tuner 'rs', Hyberband tuner 'hb' or BasyienOptimization tuner 'bo')  
    def select_tuner(self):
        
        if(self.hp_tuner_type == 'rs'):
            tuner = kt.RandomSearch(
                                        self.model,
                                        objective=self.objective,
                                        max_trials=self.max_trials,
                                        overwrite=True,
                                        directory=self.logdir_tuner,
                                        project_name="keras-rs-tuner",
                                    )
        elif(self.hp_tuner_type == 'hb'):
            tuner = kt.Hyperband(
                                    self.model,
                                    objective=self.objective,
                                    max_epochs=self.epochs,
                                    overwrite=True,
                                    directory=self.logdir_tuner,
                                    project_name="keras-hb-tuner",
                                ) 
        else:
            tuner = kt.BayesianOptimization(
                                                self.model,
                                                objective=self.objective,
                                                max_trials=self.max_trials,
                                                overwrite=True,
                                                directory=self.logdir_tuner,
                                                project_name="keras-bo-tuner",
                                            ) 
        return tuner
    
    # Classification model function contains all layers and its configurations with hyperparameter tuning in Functional API format
    def classification_model(self, hp):
        
        # Input shape
        inputs = Input(shape = self.X_train.shape[1])
        # Hidden Layers
        hidden = inputs
        # Hyperparameter Dense Layers
        hp_layers = hp.Int("Dense_layers", min_value=self.hp_layers[0], max_value=self.hp_layers[1], step=1)
        # Hyperparameter Dropout Layer
        hp_dropout = hp.Choice("Dropout_layer", self.hp_dropout_rate)
        
        # Tune each layer with number of neurons per layer 'units' and tune its activation function
        for i in range(hp_layers):
            hidden = layers.Dense(hp.Choice("dense_units_" + str(i), self.hp_units),
                                  activation=hp.Choice("activation_" + str(i), self.hp_activation))(hidden)
            hidden = layers.Dropout(hp_dropout)(hidden)
            
        # Output Layer    
        outputs = layers.Dense(pd.Series(self.y_train).unique().size, activation="softmax")(hidden)
        
        # Model Creations from input and output layers
        model = tf.keras.Model(inputs, outputs)
        
        # Model compilation with optimizer function and tune the learning rate values
        model.compile(
                        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", values=self.hp_learning_rate)),
                        loss="categorical_crossentropy",
                        metrics=["accuracy"]
                    )
        # Finally return the model
        return model
    
    # Regression model function contains all layers and its configurations with hyperparameter tuning in Functional API format
    def regression_model(self, hp):
        # Mean square logarithmic error metric to evaluate regression model
        msle = MeanSquaredLogarithmicError()
        # Input shape
        inputs = Input(shape = self.X_train.shape[1])
        # Hidden layers
        hidden = inputs
        # Hyperparameter Dense Layers
        hp_layers = hp.Int("Dense_layers", min_value=self.hp_layers[0], max_value=self.hp_layers[1], step=1)
        # Hyperparameter Dropout Layer
        hp_dropout = hp.Choice("Dropout_layer", self.hp_dropout_rate)
        
        # Tune each layer with number of neurons per layer 'units' and tune its activation function
        for i in range(hp_layers):
            hidden = layers.Dense(hp.Choice("dense_units_" + str(i), self.hp_units),
                                  activation=hp.Choice("activation_" + str(i), self.hp_activation))(hidden)
            hidden = layers.Dropout(hp_dropout)(hidden)
            
        # Output Layer     
        outputs = layers.Dense(1, activation="relu")(hidden)
        
        # Model Creations from input and output layers
        model = tf.keras.Model(inputs, outputs)
        
        # Model compilation with optimizer function and tune the learning rate values
        model.compile(
                        optimizer=tf.keras.optimizers.Adam( hp.Choice("learning_rate", values=self.hp_learning_rate)),
                        loss=msle,
                        metrics=[msle]
                    )
        # Finally return the model
        return model
    
    # Classification Tuner function to tune the classification model with the chosen tuner type and get the best model to make a prediction
    def classification_tuner(self):
        # Use LabelEncoder for categorical target
        self.y = LabelEncoder().fit_transform(self.y)
        # Use OneHotEncoder for features 
        self.X = pd.get_dummies(self.X, drop_first=True)
        # Split data into 70% train and 30% test 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, stratify=self.y,random_state=42)
        # Get log directory
        self.get_logdir()
        # Get the model
        self.model = self.classification_model
        # Evaluation metric for Classification
        self.objective = 'val_accuracy'
        # Get the chosen tuner type
        tuner = self.select_tuner()
        # Display a summary of the search space, You should see information about the three hyperparameters that we defined earlier
        tuner.search_space_summary()
        # After defining the search space, we need to select a tuner class to run the search
        tuner.search(
                        self.X_train, 
                        to_categorical(self.y_train), 
                        epochs=self.epochs, 
                        validation_split=0.2,
                        callbacks=[tf.keras.callbacks.TensorBoard(self.logdir_tensorboard)]
                    )
        # Get best model from tuner
        model = tuner.get_best_models(num_models=1)[0]
        # Make predictions
        self.y_pred = np.argmax(model.predict(self.X_test), axis=-1)
        # Get the Accuracy evaluation metric for the model
        accuracy = np.sum(self.y_pred == self.y_test) / len(self.y_test) * 100
        print("accuracy = ", accuracy)
    
    # Regression Tuner function to tune the regression model with the chosen tuner type and get the best model to make a prediction
    def regression_tuner(self):
        # Evaluation metric used
        msle = MeanSquaredLogarithmicError()
        # Use OneHotEncoder for features 
        self.X = pd.get_dummies(self.X, drop_first=True)
        # Split data into 70% train and 30% test 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        # Get log directory
        self.get_logdir()
        # get the model
        self.model = self.regression_model
        # Evaluation metric for regression
        self.objective = 'val_mean_squared_logarithmic_error'
        # Get the chosen tuner type
        tuner = self.select_tuner()
        # Display a summary of the search space, You should see information about the three hyperparameters that we defined earlier
        tuner.search_space_summary()
        # After defining the search space, we need to select a tuner class to run the search
        tuner.search(
                        self.X_train, 
                        self.y_train, 
                        epochs=self.epochs, 
                        validation_split=0.2,
                        callbacks=[tf.keras.callbacks.TensorBoard(self.logdir_tensorboard)]
                    )
        # Get best model from tuner
        model = tuner.get_best_models(num_models=1)[0]
        # Get the MSLE evaluation metric for the model
        print("msle = ",msle(self.y_test, model.predict(self.X_test)).numpy())
        
# Main Function main.py 
def main(argv):
    # Envairoment variable as default values
    dataset = 'Iris.csv'
    # Flag to chech if the problem is a classification or regression one.
    is_classification = True
    # Number of epochs
    epochs = 10
    # Tuner type
    hp_tuner_type = 'hb'
    # Nubmer of dense layers
    hp_layers = [1, 3] 
    # Activation function used
    hp_activation = ['relu', 'sigmoid', 'tanh']
    # Number of neurons or units per each layer
    hp_units = [16, 32, 64]
    # Dropout rate after each layer
    hp_dropout_rate = [0.1, 0.2, 0.3]
    # Learning rate
    hp_learning_rate = [1e-2, 1e-3, 1e-4]
    
    # Get Options of the above hyperparameters
    try:
        opts, args = getopt.getopt(argv,"hc:i:e:t:l:a:u:d:r",["is_classification=","dataset=","epochs=",
                                                              "tuner=","layersRange=","activation=",
                                                              "units=","dropout=","learningRate="])
    except getopt.GetoptError:
        print('Hyperparameters_Tuning_Script.py \n\
              -c <is_classification> \n\
              -i <dataset> \n\
              -e <epochs> \n\
              -t <tuner> [hb, rs or bo] \n\
              -l <layersRange>\n\
              -a <activation>\n\
              -u <units> \n\
              -d <dropout> \n\
              -r <learningRate>')
        sys.exit(2)
    
    # Make options Flags at which User type it and insert the arguments and values attached per each
    for opt, arg in opts:
        # Use '-h' For 'help', to see the menu and configure your specified problem
        if opt == '-h':
            print('Hyperparameters_Tuning_Script.py \n\
                  -c <is_classification> \n\
                  -i <dataset> \n\
                  -e <epochs> \n\
                  -t <tuner> [hb, rs or bo] \n\
                  -l <layersRange>\n\
                  -a <activation>\n\
                  -u <units> \n\
                  -d <dropout> \n\
                  -r <learningRate>')
            sys.exit()
            
        # Use '-c' For 'classification', to choose if the problem is a classification or regression    
        elif opt in ("-c", "--is_classification"):
            is_classification = int(arg)
        
        # Use '-i' For 'input', to insert the dataset 
        elif opt in ("-i", "--dataset"):
            dataset = arg
        
        # Use '-e' For 'epochs', to choose epochs number
        elif opt in ("-e", "--epochs"):
            epochs = arg
        
        # Use '-t' For 'tuner', to choose the tuner type
        elif opt in ("-t", "--tuner"):
            hp_tuner_type = arg
        
        # Use '-l' For 'layer range', to select the range of layers used
        elif opt in ("-l", "--layersRange"):
            hp_layers = list(map(int, arg.split(',')))
        
        # Use '-a' for 'activations', to select the activation function used for each layer
        elif opt in ("-a", "--activation"):
            hp_activation = arg.split(',')
        
        # Use '-u' For 'units', to select units nubmers per layer
        elif opt in ("-u", "--units"):
            hp_units = list(map(int, arg.split(',')))
        
        # Use '-d' For 'dropout', to insert the dropout rate
        elif opt in ("-d", "--dropout"):
            hp_dropout_rate = list(map(float, arg.split(',')))
        
        # Use '-r' for 'learning rate', to choose learning rate
        elif opt in ("-r", "--learningRate"):
            hp_learning_rate = list(map(float, arg.split(',')))
    
    # Call the class with specified entered parameters
    hpo = HPO(dataset, epochs, hp_tuner_type, hp_layers, hp_activation, hp_units, hp_dropout_rate, hp_learning_rate)
    
    # Chech the dataset is a Classification or Regression
    if is_classification:
        hpo.classification_tuner()
    else:
        hpo.regression_tuner()


# Main with all argument vectors passed         
if __name__ == "__main__":
    main(sys.argv[1:])

#
# End of Script
#
        
'''
Important Tips: 

To run this script, you just need to type "python scriptName.py" in your cmd at the destination of the same file location.
To get the TensorBoard visualizaation, you also just need to type "tensorboard --logdir" + TensorBoard path and copy the URL appears in web page. 

'''  

#______________________________________________________________________________________________________________________________________________________
'''
(base) C:\Users\Lamyaa Zayed>cd FreeLance_Task
(base) C:\Users\Lamyaa Zayed\FreeLance_Task>python Hyperparameters_Tuning_Script.py
(base) C:\Users\Lamyaa Zayed\FreeLance_Task>python Hyperparameters_Tuning_Script.py -h
(base) C:\Users\Lamyaa Zayed\FreeLance_Task>tensorboard --logdir \logs\20220424-022929\TensorBoard
http://localhost:6006/  -->> copy this url to web page
'''