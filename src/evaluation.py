
import json
import os
os.system('pip install nltk')
os.system('pip install torch==1.13.0')
os.system('pip install torchtext==0.14.0')
import torch
from torch import nn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download("punkt")
nltk.download('wordnet')
import argparse
import tarfile
from torch.utils.data import Dataset

class sentiment(nn.Module):
    # Initializing the model
    def __init__(self, vocab_len, embed_dim, lstm_size, bidirectional, num_layers, dropout = 0):
        super().__init__()
        # Saving necessary parameters
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.std_out = []
        # Embedding layer. Expects indices between 0 and vocab_len and generates vectors of embed_dim 
        # for each index. Also assigns 0 to the padding index vector.
        # Input of shape (batch_size, sequence_len)
        self.embeddings = nn.Embedding(vocab_len, embed_dim, padding_idx = 0)
        # LSTM layer. Expects input of shape (batch_size, sequence_len, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size = lstm_size, batch_first = True, bidirectional = bidirectional, dropout = dropout)
        
        # Determining the size of the linear layer based on the biderectionality of LSTM
        if self.bidirectional:
            self.linear_size = lstm_size * 2
        else:
            self.linear_size = lstm_size
        
        # Linear layer. Expects inputs of shape (batch_size, linear_size)
        self.linear = nn.Linear(self.linear_size, 1)
        # The sigmoid layer
        self.sigmoid = nn.Sigmoid()
    
    # Defining the forward pass
    def forward(self, inputs):
        
        # Running the input sequence through the embedding layer and lstm layer
        x = self.embeddings(inputs)
        x = self.lstm(x)[1][0]
        
        # Determining which part of the LSTM output to use based on LSTM structure
        if self.bidirectional:
            x = x[-2 :, :, :]
            x = x.permute(1,0,2).reshape(x.size(1),-1)
        else:
            x = x[-1, :, :]
            x = x.reshape(x.size(0),-1)
        # PAssing the LSTM output through a linear and a softmax layer
        x = self.linear(x)
        return x[:,0]
    
    

def model_fn(model_dir):
    vocabulary = torch.load(model_dir + "/vocabulary.pth")

    with open(model_dir+ "/model_info.json") as file:
        model_info = json.load(file)
    
    model_name = "model.pth"
    
    model_params = torch.load(model_dir + "/" + model_name)
    
    # Size of the embedding vector for each token
    embed_dim = int(model_info['embed_dim'])
    # Size of the lstm output
    lstm_size = int(model_info["lstm_size"])
    # Whether to run a bidirectional LSTM
    bidirectional = bool(model_info["bidirectional"])
    # Number of LSTM layers
    num_layers = int(model_info["num_layers"])
    
    model = sentiment(len(vocabulary), embed_dim, lstm_size, bidirectional, num_layers)
    
    model.load_state_dict(model_params)
    
    return model

def evaluate(model, test_dataset, threshold):
    """
    Evaluates the classification of the model based on the test set. 

    Parameters
    ----------
    test_dataset : obj
        PyTorch test Dataset.

    Returns
    -------
    report : dict
        Classification evaluation report.

    """
    with torch.no_grad():
        model.eval()
        predictions = model(test_dataset[:, :-1])
        predictions_arg = torch.where(predictions >= threshold, torch.tensor(1), torch.tensor(0))
        accuracy = torch.sum(predictions_arg == test_dataset[:, -1]).item() / len(test_dataset) * 100
        return accuracy         
        
class dataset(Dataset):
    """
    Pytorch Dataset class for converting the pandas dataframes into datasets
    """
    def __init__(self,data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    model_dir = "/opt/ml/processing/model/"
    model_tar_path = model_dir + 'model.tar.gz'
                  
    with tarfile.open(model_tar_path) as tar_f:
        tar_f.extractall(model_dir)
    
    
    model = model_fn(model_dir)
    
    test_dataset = torch.load("/opt/ml/processing/data/test_dataset.pth")
    
    accuracy = evaluate(model, test_dataset, 0.5)
    
    report_dict = {
     "metrics": {
         "accuracy": {
             "value": accuracy,
                     },
                 },
             }
    
    evaluation_path = '/opt/ml/processing/output/evaluation.json'
       
    with open(evaluation_path, "w") as f:
        json.dump(report_dict, f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    