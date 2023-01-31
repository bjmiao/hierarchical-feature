# import Neural_Decoding
# from Neural_Decoding.decoders import WienerFilterDecoder, SVRDecoder, WienerFilterClassification, DenseNNClassification
from sklearn.linear_model import LogisticRegression #For Wiener Filter and Wiener Cascade

import numpy as np

try:
    import keras
    keras_v1=int(keras.__version__[0])<=1
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
    from keras.utils import np_utils
    
except ImportError:
    print("\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders")
    pass
from tensorflow.keras import layers
from tensorflow.keras import regularizers
try:
    import keras
    keras_v1=int(keras.__version__[0])<=1
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
    from keras.utils import np_utils
    from keras.callbacks import CSVLogger
    
except ImportError:
    print("\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders")
    pass
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from sklearn import linear_model
class LinearClassification(object):

    """
    Class for the Wiener Filter Decoder
    There are no parameters to set.
    This simply leverages the scikit-learn logistic regression.
    """

    def __init__(self,C=1):
        self.C=C
        return

    def fit(self,X_flat_train,y_train):
        """
        Train Wiener Filter Decoder
        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly
        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        # if self.C>0:
        self.model=linear_model.LogisticRegression(C=self.C,multi_class='auto') #Initialize linear regression model
        # else:
            # self.model=linear_model.LogisticRegression(penalty='none',solver='newton-cg') #Initialize linear regression model
        self.model.fit(X_flat_train, y_train) #Train the model
    
    def dump_coef(self, filename):
        np.save(filename, self.model.coef_)

    def predict(self,X_flat_test):
        """
        Predict outcomes using trained Wiener Cascade Decoder
        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.
        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """
        y_test_predicted=self.model.predict(X_flat_test) #Make predictions
        return y_test_predicted

class RandomForestClassification(object):
    def __init__(self):
        self.clf = RandomForestClassifier(random_state = 0)
    def fit(self, X_flat_train,y_train, X_val, y_val):
        self.clf.fit(X_flat_train, y_train)
    def predict(self, X_flat_test):
        y_test_predicted = self.clf.predict(X_flat_test)
        return y_test_predicted

class DenseNNClassification(object):

    """
    Class for the dense (fully-connected) neural network decoder
    Parameters
    ----------
    units: integer or vector of integers, optional, default 400
        This is the number of hidden units in each layer
        If you want a single layer, input an integer (e.g. units=400 will give you a single hidden layer with 400 units)
        If you want multiple layers, input a vector (e.g. units=[400,200]) will give you 2 hidden layers with 400 and 200 units, repsectively.
        The vector can either be a list or an array
    dropout: decimal, optional, default 0
        Proportion of units that get dropped out
    num_epochs: integer, optional, default 10
        Number of epochs used for training
    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0,logfile="training.log"):
         self.dropout = dropout
         self.num_epochs = num_epochs
         self.verbose = verbose
         self.logfile = logfile

         #If "units" is an integer, put it in the form of a vector
         try: #Check if it's a vector
             units[0]
         except: #If it's not a vector, create a vector of the number of units for each layer
             units=[units]
         self.units=units

         #Determine the number of hidden layers (based on "units" that the user entered)
         self.num_layers=len(units)

    def fit(self,X_flat_train,y_train, X_val, y_val):

        """
        Train DenseNN Decoder
        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly
        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        y_total = np.append(y_train, y_val)
        #Use one-hot coding for y
        if y_train.ndim==1:
            y_train=np_utils.to_categorical(y_train.astype(int), num_classes = max(y_total) + 1)
        elif y_train.shape[1]==1:
            y_train=np_utils.to_categorical(y_train.astype(int), num_classes = max(y_total) + 1)

        if y_val.ndim==1:
            y_val=np_utils.to_categorical(y_val.astype(int), num_classes = max(y_total) + 1)
        elif y_val.shape[1]==1:
            y_val=np_utils.to_categorical(y_val.astype(int), num_classes = max(y_total) + 1)

        model=Sequential() #Declare model
        #Add first hidden layer
        # model.add(Dense(self.units[0],input_dim=X_flat_train.shape[1])) #Add dense layer
        model.add(Dense(self.units[0],input_dim=X_flat_train.shape[1], )) #Add dense layer
        model.add(Activation('relu')) #Add nonlinear (tanh) activation
        # if self.dropout!=0:
        if self.dropout!=0: model.add(Dropout(self.dropout))  #Dropout some units if proportion of dropout != 0

        #Add any additional hidden layers (beyond the 1st)
        for layer in range(self.num_layers-1): #Loop through additional layers
            model.add(Dense(self.units[layer+1])) #Add dense layer
            model.add(Activation('tanh')) #Add nonlinear (tanh) activation - can also make relu
            if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units if proportion of dropout != 0

        #Add dense connections to all outputs
        model.add(Dense(y_train.shape[1])) #Add final dense layer (connected to outputs)
        model.add(Activation('softplus'))

        #Fit model (and set fitting parameters)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) #Set loss function and optimizer
        csv_logger = CSVLogger(self.logfile, append=True)
        if keras_v1:
            history = model.fit(X_flat_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose, callbacks=[csv_logger]) #Fit the model
        else:
            history = model.fit(X_flat_train,y_train,epochs=self.num_epochs,
                validation_data=(X_val, y_val),
                verbose=self.verbose, callbacks=[csv_logger]) #Fit the model
            self.model=model
            self.history = history

    def predict(self,X_flat_test):

        """
        Predict outcomes using trained DenseNN Decoder
        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.
        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted_raw = self.model.predict(X_flat_test) #Make predictions

        y_test_predicted=np.argmax(y_test_predicted_raw,axis=1)

        return y_test_predicted