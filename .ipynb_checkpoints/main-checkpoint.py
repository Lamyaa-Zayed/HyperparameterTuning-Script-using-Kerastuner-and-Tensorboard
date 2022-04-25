import sys
import tensorboard
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from sklearn import datasets
from sklearn.model_selection import train_test_split
#
# Import Keras modules
#
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
#
# Create LogDir
#
from datetime import datetime

class HPO():
    def __init__(self, dataset, epochs):
        self.epochs = epochs
        self.df = pd.read_csv(dataset)
        self.X = np.array(self.df)[:,:-1]
        self.y = np.array(self.df)[:,-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, 
                                                                                stratify=y,random_state=42)
    def get_logdir():
        now = datetime.now()
        self.logdir_tuner = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/" + "Tuner"
        self.logdir_tensorboard = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/" + "TensorBoard"

    def classification_model(hp):
        inputs = Input(shape = self.X_train.shape[1])
        hidden = inputs

        hp_layers = hp.Int("Dense_layers", min_value=1, max_value=3, step=1, default=3)
        hp_dropout = hp.Choice("Dropout_layer", [0.1, 0.2, 0.3], default=0.2)

        for i in range(hp_layers):
            hidden = layers.Dense(hp.Choice("dense_units_" + str(i), [16, 32, 64], default=16),
                                  activation=hp.Choice("activation_" + str(i), ['relu', 'tanh']))(hidden)
            hidden = layers.Dropout(hp_dropout)(hidden)

        outputs = layers.Dense(pd.Series(y_train).unique().size, 
                               activation="softmax")(hidden)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice("learning_rate", values=[1e-2, 1e-3])
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def classification_tuner():
        tuner = kt.RandomSearch(
                                    self.classification_model,
                                    objective="val_accuracy",
                                    max_trials=10,
                                    overwrite=True,
                                    directory=self.logdir_tuner,
                                    project_name="keras-tuner",
                                )
        tuner.search_space_summary()
        tuner.search(
                        self.X_train[:,:], 
                        to_categorical(self.y_train), 
                        epochs=self.epochs, 
                        validation_data=(self.X_test[:,:], to_categorical(self.y_test)),
                        callbacks=[tf.keras.callbacks.TensorBoard(self.logdir_tensorboard)]
                    )
        model = tuner.get_best_models(num_models=1)[0]
        y_pred = np.argmax(model.predict(self.X_test), axis=-1)
        accuracy = np.sum(self.y_pred == self.y_test) / len(self.y_test) * 100
        return accuracy
     
# main.py
if __name__ == "__main__":
    
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")
        
    