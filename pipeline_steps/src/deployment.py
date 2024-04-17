import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import json
import os
import torch
from torch import nn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download("punkt")
nltk.download('wordnet')


def cleanup_text(text):
    """
    Performs the following tasks on the text:
        - lowercasing all the characters
        - removing non-alphabet characters excluding "., !, (, ), \n, :, ?"
        - removing any multiple consecutive occurence of the excluded characters above
    
    Parameters
    ----------
    text : str
        text to be cleaned.

    Returns
    -------
    text : str
        cleaned text.
    """
    
    text = text.lower()
    text = re.sub(r"[^a-z.?!:)( \n]+", "", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\.{2,}", "!", text)
    text = re.sub(r"\.{2,}", "?", text)
    text = re.sub(r"\.{2,}", ")", text)
    text = re.sub(r"\.{2,}", "(", text)
    text = re.sub(r"\.{2,}", ":", text)
    return text



def process_reviews(review, tokenizer, lemmatizer, vocabulary, max_len):
    """
    Performs the following tasks on each review text:
        - cleaning the text
        - tokenizing the text
        - lemmantizing the text
        - converting the tokens into indices 
        - padding and truncating the review based on max_len passed to the function

    Parameters
    ----------
    review : str
        The product review text.
    tokenizer : obj
        Tokenizer object for tokenization of the text.
    lemmatizer : obj
        Lemmantizer onject for Llmmantization of the text.
    vocabulary : obj
        Vocabulary object correspoding tokens and indices.
    max_len : int
        Maximum allowed length of a product review.
    pad : boolean
        Either to pad the input or not.

    Returns
    -------
    review_processed : list
        A list of indices.
    """
    
    review_cleaned = cleanup_text(review)
    review_tokenized = tokenizer(review_cleaned)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in review_tokenized]
    review_processed = vocabulary(lemmatized_text)

    if len(review_processed) > max_len:
        review_processed = review_processed[:max_len]
    return review_processed


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
    
    return (vocabulary, model_info, model)

def predict_fn(input_data, model):

    vocabulary, model_info, model = model
    
    model.eval()
    
    review_processed = process_reviews(input_data, word_tokenize, WordNetLemmatizer(), vocabulary, len(vocabulary))
    
    threshold = float(model_info["threshold"])
    
    with torch.no_grad():
        prediction = torch.where(model.sigmoid(model(torch.tensor(review_processed).reshape(1, -1))) >= threshold, torch.tensor(1), torch.tensor(0))

        if prediction == 1:
            return "Positive"
        else:
            return "Negative"

def input_fn(serialized_input_data, content_type='application/jsonlines'): 

    input_data = json.loads(serialized_input_data)
    return input_data["input_text"]

def output_fn(prediction_output, content_type):
    return json.dumps(prediction_output)

