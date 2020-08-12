"""
    nn
    ~~~
    
    Simple NN to run in Flask
    
"""
from flask import Flask, request
import ast
import traceback
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


app = Flask(__name__) # create the Flask app

seed = 78
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

class Perceptron(nn.Module):
    """ A Perceptron has one Linear layer """

    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): size of the input features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """The forward pass of Perceptron

        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, input_dim)
        Returns:
            the resulting tensor. tensor.shape should be (batch, 1)
        """
        return torch.sigmoid(self.fc1(x_in))

     


def train(X, y, model, optimizer_fun, loss_fun, n_epochs, change=1.0, epsilon=1e-3):
    """Train model untill we get the required loss value. Save model parameters.
    
    Args:
        X - input data, torch.Size([n, 3])
        y - targets, torch.Size([n])
        model - model to train (Perceptron)
        optimizer_fun - optimizer function
        loss_fun - loss function
        n_epochs - number of epochs      
    Returns:
        trained model, training set predictions as a list of '0' and '1'-s, 
        last loss, all losses from all iterations
    """
    losses = []
    last = 10.0
    epoch = 0
    while change > epsilon or epoch < n_epochs or last > 0.3:
        optimizer_fun.zero_grad()
        yhat = model(X).squeeze()
        loss = loss_fun(yhat, y)
        loss.backward()
        optimizer_fun.step()
               
        loss_value = loss.item()
        losses.append(loss_value)

        change = abs(last - loss_value)
        last = loss_value
        #print("epoch: {} loss: {} change: {}".format(epoch, loss_value, change))
        
        epoch += 1
        
    classes = [ int(y.detach().numpy() > 0.5) for y in yhat]
    return model, classes, last, losses

# Model parameters

n_epochs = 15
input_dim = 3
#trained_model = None #Perceptron(input_dim=input_dim)
loss_fun = nn.BCELoss()
MODEL_PATH = "./simple-model"

@app.route('/train', methods=['GET', 'POST']) #allow both GET and POST requests
def get_lr_and_train():
    """ Get learning rate and train model """

    # Training set
    X = torch.tensor([[0,0,1], [1,1,0], [1,0,1], [0,1,1]], dtype=torch.float32)
    y = torch.tensor([0,1,1,0], dtype=torch.float32)

    lr_min = 0.01
    lr_max = 0.02
    lr = 0.01
    msg = ''
    header = "<h3>Train model and get predictions on the traing set </h3>"
    lr_format = "<br>Learning rate must be in range: {} <= lr <= {}".format(lr_min, lr_max)
    
    if request.method == 'POST':  # process submission
        model = Perceptron(input_dim=input_dim)
        optimizer_fun = optim.Adam(params=model.parameters(), lr=lr)
        lr_str = request.form.get('lr')
        
        try:
            lr_new = ast.literal_eval(lr_str)
            assert lr_new >= lr_min and lr_new <= lr_max
            lr = lr_new
        except Exception as e:
            #logging.error(traceback.format_exc())
            msg="Incorrect input: {}".format(lr_str)+lr_format
        else:
            pass
            #msg="New learning rate: {}".format(lr)
        finally:           
            
            model, classes, last, losses = train(X, y, model, optimizer_fun, 
                                  loss_fun, n_epochs, change=1.0, epsilon=1e-3)
            
            # Save model parameters
            torch.save(model.state_dict(), MODEL_PATH)
            
            msg_train = "<p><h4>Last loss: {} <br>Classes in train set: {} </h4>".format(last, classes)
            msg = header+msg +"<br>Training with learnin rate: {} ...".format(lr)+msg_train


        return msg

    return header+lr_format+'''<form method="POST">
                  Enter learning rate: <input type="text" name="lr"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''



def predict(model, X):
    """Predict classes (labels) of X
    
    Args:
        model - trained model (Perceptron)
        X - data to predict class, torch.Size([n, 3])
    Returns:
        predictions as a list of '0' and '1'-s, 
    """
    yhat = model(X).squeeze()
    yhat = yhat.detach().numpy()
    if (yhat.shape == np.array(0).shape): # for a single prediction
        yhat = [yhat.item(0)]
    classes = [int(y > 0.5) for y in yhat]
    return classes    

@app.route('/predict', methods=['GET', 'POST']) #allow both GET and POST requests
def get_x_and_predict():
    header = "<h3>Predict classes for new data </h3> "
    data_format = """Input format: [[x11,x12,x13],[x21,x22,x23], ...[xn1,xn2,xn3]],
                    <br>where x = {0,1}, for example [[1,0,0], [1,1,1], ...] """
    msg = ''

    if request.method == 'POST':  # process submission
        X_str = request.form.get('X')
 
        try:
            # Load model parameters
            model = Perceptron(input_dim=input_dim)
            model_state_dict = torch.load(MODEL_PATH)
            model.load_state_dict(model_state_dict)
            
            l = ast.literal_eval(X_str)
            x_input = torch.tensor(l, dtype=torch.float32)
            assert(x_input.shape[1] == 3)
            
        except Exception as e:
            #logging.error(traceback.format_exc())
            msg = X_str+" - Incorrect input, please use: <br>"+data_format
        else:
            classes = predict(model, x_input)
            msg="Input data: {} <p><h4>Predictions: {}</h4>".format(x_input, classes)
        finally:        
            msg = header+msg

        return msg

    return header+data_format+'''<form method="POST">
                  <p>Enter X data: <input type="text" name="X"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5078) 

