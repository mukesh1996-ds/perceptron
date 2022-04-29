import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from matplotlib.colors import ListedColormap

class preceptron:
    def __init__(self, eta: float=None, epochs: int=None):  # constructor class step = 1
    # eta = learning rate , epochs = iteration
        self.weights = np.random.randn(3) * 1e-4  # small random weights in used
        training = (eta is not None) and (epochs is not None)
        if training:
            print(f"Initial weight before training: \n {self.weights}")
        self.eta = eta
        self.epochs = epochs
    
    def _z_outcome(self, inputs, weights): # internal method step = 3
        return np.dot(inputs, weights) # output (wx + b) 
    
    def activation_function (self, z):  # activation applied on hidden layer step 4
        return np.where(z > 0, 1, 0)
    
    def fit (self, x, y):  # Fitting the model step = 2
        self.x = x
        self.y = y
        
        x_with_bias = np.c_[self.x, -np.ones((len(self.x), 1))]  # matrix of x with bias 
        # applying concation operation
        print(f"x with bias : \n {x_with_bias}")
        # multiple iteration 
        for epoch in range(self.epochs):
            print(f"for epoch >> {epoch}")
            
            z = self._z_outcome(x_with_bias,self.weights)
            y_hat = self.activation_function(z)
            print(f"predicted value after forward pass: \n{y_hat}")
            
            self.error = self.y - y_hat
            print(f"error: \n {self.error}")
            
            self.weights = self.weights + self.eta * np.dot(x_with_bias.T, self.error)  # weight update rule
            print(f"updated weights after epoch: \n {epoch}/{self.epochs}: \n{self.weights}")

            
    def predict (self,x_test):  # predicting the new data step = 5
        x_with_bias = np.c_[x_test, -np.ones((len(x_test), 1))]
        z = self._z_outcome(x_with_bias,self.weights)
        return self.activation_function(z)
    
    def total_loss(self):  # loss calculation  step = 6
        total_loss = np.sum(self.error)
        print(f"total loss : \n {total_loss}") 
        
    def _create_dir_return_path(self, model_dir, filename):  # internal method : directory path step 8
        os.makedirs(model_dir, exist_ok = True )
        return os.path.join(model_dir, filename)
        
    def save(self, filename, model_dir=None): # saving the model step = 7 
        if model_dir is not None:
            model_file_path= self._create_dir_return_path(model_dir, filename)
            joblib.dump(self, model_file_path)
        else:
            model_file_path = self._create_dir_return_path("model",filename)
            joblib.dump(self,model_file_path)
            
    def load_model(self, filepath):  # reloading the model stpe 9 
        return joblib.load(filepath)
    

# Data preparation
def prepare_data(df, target_column = 'y'):
    x = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return x,y


# Defining a data set i.e. AND gate

AND = {"x1": [0,0,1,1], "x2": [0,1,0,1], "y": [0,0,0,1]}


# converting it into data frame

df_AND = pd.DataFrame(AND)

print(f"the data set look's like : \n {df_AND}")


# perceptron implementation
x,y = prepare_data(df_AND)

eta = 0.1  # 0 and 1
epochs = 10

model_and = preceptron(eta = eta, epochs = epochs)
model_and.fit(x,y)
_ = model_and.total_loss()


OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y" : [0,1,1,1]
}

df_OR = pd.DataFrame(OR)

df_OR

X, y = prepare_data(df_OR)

ETA = 0.1 # 0 and 1
EPOCHS = 10

model_or = preceptron(eta=ETA, epochs=EPOCHS)
model_or.fit(X, y)

_ = model_or.total_loss()


model_or.save(filename="or.model", model_dir="model_or")
 