import numpy as np
from crossentropy_batch import cross_entropy_vectorized
from feedforward import forward_batch
from backprop import backward
from data_helper import load_data
from sklearn.metrics import accuracy_score
import os
import json

def save_json_with_numpy(data, filename):
    def numpy_to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: numpy_to_json(v) for k, v in obj.items()}
        else:
            return obj

    try:
        # Check if the file exists
        if os.path.exists(filename):
            # Read existing data
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Append new data to existing data
            existing_data[f'trial_{len(existing_data)}'] = data
            
            # Write updated data back to the file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, default=numpy_to_json, ensure_ascii=False, indent=4)
            
            print(f"New data appended to {filename}")
        else:
            # If file doesn't exist, create it with the new data
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({'trial_0': data}, f, default=numpy_to_json, ensure_ascii=False, indent=4)
            
            print(f"File {filename} created with new data")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file '{filename}'. Starting fresh.")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({'trial_0': data}, f, default=numpy_to_json, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




class NeuralNetworkMultiClassifier:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, learing_rate = 1e-2, batch_size = 32, n_epochs = 20):
        self.W1 = np.random.randn(input_dim, hidden_dim1)
        self.b1 = np.zeros((1, hidden_dim1))
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2)
        self.b2 = np.zeros((1, hidden_dim2))
        self.W3 = np.random.randn(hidden_dim2, output_dim)
        self.b3 = np.zeros((1, output_dim))
        self.learning_rate = learing_rate
        self.batch_size = batch_size
        self.epochs = n_epochs
        
    def update_weights(self,dW1, db1, dW2, db2, dW3, db3,learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
    def get_loss_test(self,X,Y):
        _, _, y_pred = forward_batch(X, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        test_loss = cross_entropy_vectorized(Y,y_pred)
        accuracy = accuracy_score(np.argmax(Y, axis = 1), np.argmax(y_pred,axis = 1))
        return test_loss, accuracy


    def train(self, X_train, y_train):
        n_epochs = self.epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        
        for epoch in range(n_epochs):
            for batch in range((X_train.shape[0] + batch_size - 1) // batch_size):
                begin = batch * batch_size
                end = min((batch + 1) * batch_size, X_train.shape[0])
                inputs = X_train[begin:end]
                targets = y_train[begin:end]
                out1, out2, y_pred = forward_batch(inputs, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
                loss = cross_entropy_vectorized(targets,y_pred)
                dW1, db1, dW2, db2, dW3, db3 = backward(inputs, targets, self.W3, self.W2, out1, out2, y_pred)
                self.update_weights(dW1, db1, dW2, db2, dW3, db3, learning_rate)

                print(f"In epoch {epoch}, batch {batch}: Loss = {loss:.4f}")

            
def model_metadata():
    data = {

    }
def normalize(X):
    X = X / 255
    return X
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    X_train_n, X_test_n = normalize(X_train), normalize(X_test)
    filename = 'model_metadata.json'
    model = NeuralNetworkMultiClassifier(784, 20, 15, 10,1e-3,64,40)
    model.train(X_train_n, y_train)
    loss, accuracy = model.get_loss_test(X_test_n, y_test)

    data = {'loss': loss,
            'accuracy': accuracy,
            'model_weights' : [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3],
            'batch_size' : model.batch_size,
            'learning_rate': model.learning_rate,
            'number_of_epochs': model.epochs
            }

    save_json_with_numpy(data, filename)
    