import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
# import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time
from datetime import datetime
os.system('pip install torchtext==0.14.0')
# import torchtext
import json
import matplotlib.pyplot as plt
import textwrap
#------------------------------------------------------------------------------
# Function Definitions


# Defining the Model
class sentiment(nn.Module):
    
    # Initializing the model
    def __init__(self, vocab_len, embed_dim = 20, lstm_size = 20, bidirectional = True, num_layers = 1, dropout = 0):
        super().__init__()
        # Saving necessary parameters
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.training_info = []
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
    
    def train_(self, train_dataloader, validation_dataset, epochs, loss_func, optimizer, device = torch.device("cpu"),threshold = 0.5):
        """
        Runs the training loop for of the model and trains the model based on the input parameters.

        Parameters
        ----------
        train_dataloader : obj
            PyTorch DataLoader object for training data.
        validation_dataset : obj
            PyTorch Database object for validation evaulation.
        epochs : int
            DESCRIPTION.
        loss_func : obj
            Loss function to use.
        optimizer : obj
            Optimizer to use.

        Returns
        -------
        None
        """

        # Saving and defining the required parameters and variables
        self.loss_func = loss_func
        num_samples = len(train_dataloader.dataset)
        loss_train_list = []
        accuracy_train_list = []
        loss_validation_list = []
        accuracy_validation_list = []
        # Moving the validation dataset to GPU if available
        validation_dataset = validation_dataset[:, :].to(device)
        # loss_validatoion = 0
        
        # Running the training loop
        for epoch in range(epochs):    
            correct_sentiments = 0
            epoch_loss = 0
            for data in train_dataloader:
                data = data.to(device)
                # the last elemnt in the sequence is the sentiment and the rest are the input sequence
                review = data[:, :-1]
                sentiment_real = data[:, -1].to(torch.float)
                sentiment_pred = self.forward(review)
                sentiment_pred = self.sigmoid(sentiment_pred)
                sentiment_pred_args = torch.where(sentiment_pred >= threshold, torch.tensor(1), torch.tensor(0))
                correct_sentiments += torch.sum(sentiment_pred_args == sentiment_real).item()
                
                optimizer.zero_grad()
                loss_train = self.loss_func(sentiment_pred, sentiment_real)
                loss_train.backward()
                epoch_loss += loss_train.item()
                optimizer.step()
                break
            with torch.no_grad():
                validation_preds = self.forward(validation_dataset[:, :-1])
                loss_validatoion = self.loss_func(validation_preds, validation_dataset[:, -1].to(torch.float)).item()
            accuracy_validation_args = torch.where(validation_preds >= threshold, torch.tensor(1), torch.tensor(0))
            accuracy_validation = (torch.sum(accuracy_validation_args == validation_dataset[:, -1]).item() / len(validation_dataset))
            
            loss_train_list.append(epoch_loss/len(train_dataloader))
            accuracy_train_list.append(correct_sentiments/num_samples * 100)
            loss_validation_list.append(loss_validatoion)
            accuracy_validation_list.append(accuracy_validation * 100)
            
            self.training_info.append(textwrap.dedent(f"""
                                                 epoch : {epoch + 1}, training loss : {epoch_loss/len(train_dataloader):.4f}, training accuracy : {correct_sentiments/num_samples * 100:.1f}
                                                 epoch : {epoch + 1}, validation loss : {loss_validatoion:.4f}, validation accuracy : {accuracy_validation * 100:.1f}
                                                 """))

         
        # Plotting the training and validation accuracy and loss during the model training
        plt.figure()
        plt.plot(range(1, epochs + 1), loss_train_list, label = "training loss")
        plt.plot(range(1, epochs + 1), loss_validation_list, label = "validation loss")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Train and Validation Loss per Epoch")
        plt.savefig(args.data_output_dir + "/Train_Validation_Loss.png")
        
        plt.figure()
        plt.plot(range(1, epochs + 1), accuracy_train_list, label = "training accuracy")
        plt.plot(range(1, epochs + 1), accuracy_validation_list, label = "validatoin accuracy")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.title("Train and Validation Accuracy per Epoch")
        plt.savefig(args.data_output_dir + "/Train_Validation_Accuracy.png")
        
    def evaluate(self, test_dataset, threshold):
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
            predictions = self.forward(test_dataset[:, :-1])
            predictions_arg = torch.where(predictions >= threshold, torch.tensor(1), torch.tensor(0))
            accuracy = torch.sum(predictions_arg == test_dataset[:, -1]).item() / len(test_dataset) * 100
            loss = self.loss_func(predictions, test_dataset[:, -1].to(torch.float)).item()
            report = classification_report(test_dataset[:, -1], predictions_arg, output_dict=True, zero_division=0)
            self.training_info.append(f"test set loss : {loss:.4}, test set accuracy : {accuracy:.1f}")
        return report 


def set_to_gpu():
    """
    Sets the device to GPU if available otherwise sets it to CPU. Uses MPS if on mac and CUDA 
    otherwise.

    Returns
    -------
    device : obj.
        PyTorch device object for running the mode on GPU or CPU.
    """
    # Setting the constant seed for repeatability of results.
    seed = 10
    torch.manual_seed(seed)
    
    # Setting the device to CUDA if available
    if torch.cuda.is_available():
       device = torch.device("cuda")
       torch.cuda.manual_seed_all(seed)
       torch.cuda.empty_cache()
    # Setting the device to CPU if GPU not available
    else:
        device = torch.device("cpu")
    # Setting deterministicc behaviour for repatability of results. 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return device


def train_model(args, embed_dim = 20, lstm_size = 20, bidirectional = True,\
                num_layers = 1, dropout = 0, learning_rate = 0.001,\
                epochs = 100, threshold = 0.5, batch_size = 64):
    """
    Trains an LSTM model using the input parameters. Saves the model. Also evaluates the 
    classification accuracy on the test set and return the classification report
    besides the model object.

    Parameters
    ----------
    embed_dim : int, optional
        Size of the embedding vector for each token. The default is 20.
    lstm_size : int, optional
        Size of the lstm output. The default is 20.
    bidirectional : boolean, optional
        Whether to run a bidirectional LSTM. The default is True.
    num_layers : int, optional
        Number of LSTM layers. The default is 1.
    dropout : float, optional
        LSTM dropout. The default is 0.
    learning_rate : float, optional
        Learning rate for trianing the model. The default is 0.001.
    epochs : int, optional
        Number of epochs to run. The default is 100.

    Returns
    -------
    report : dict
        A dictionary contatining the classification report based on the test datset.
    model : oobj
        PyTorch model.

    """
    # Starting the timer to measure how long model training takes
    start_time = time.time()
    
    
    
    # Setting the device to GPU if available
    device = set_to_gpu()
    
    # Reading the vocabulary 
    vocabulary = torch.load(args.vocabulary_dir + "/vocabulary.pth")
    torch.save(vocabulary, args.model_dir + "/vocabulary.pth")

    # Reading the train, test and validation datasets
    training_dataset = torch.load(args.train_data_dir + "/training_dataset.pth")
    validation_dataset = torch.load(args.validation_data_dir + "/validation_dataset.pth")
    test_dataset = torch.load(args.test_data_dir + "/test_dataset.pth")
    # Creating a dataloader from he training dtaset for the model training loop
    train_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)    
    
    # Instansiating the model
    model = sentiment(len(vocabulary), embed_dim, lstm_size, bidirectional, num_layers, dropout).to(device)
    # Instansiating the optimizer and loss function
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    # Training the model
    model.train_(train_loader, validation_dataset, epochs, loss_func, optimizer, device, threshold)
    
    # Measuring the elapsed time and reporting it
    elapsed_time = time.time() - start_time
    model.training_info.append(f"\nTime it took to train the model: {elapsed_time:.1f}s\n")
    
    #Evaluating the model using the test set and saving the classification report
    model.to("cpu")
    report = model.evaluate(test_dataset, threshold)
    
    with open(args.data_output_dir + "/classification_report.json", "w") as file:
        json.dump(report, file)
    
    with open(args.data_output_dir + "/training_info.txt", "w") as file:
        file.write("\n".join(model.training_info))
    
    # Saving the model
    # now = datetime.now()
    # model_name = args.model_dir + "/LSTM model-" + now.strftime("%y_%m_%d-%H_%M_%S")
    torch.save(model.state_dict(), args.model_dir + '/model.pth')
    
    return report, model


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


def parse_args():
    parser = argparse.ArgumentParser(description='processing job')
    
    parser.add_argument("--embed_dim", default = 20)
    parser.add_argument("--lstm_size", default = 20)
    parser.add_argument("--bidirectional", default = True)
    parser.add_argument("--num_layers", default = 1)
    parser.add_argument("--dropout", default = 0.0)
    parser.add_argument("--learning_rate", default = 0.001)
    parser.add_argument("--epochs", default = 5)
    parser.add_argument("--threshold", default = 0.5)
    parser.add_argument("--batch_size", default = 32)
    parser.add_argument("--train_data_dir", default = os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument("--validation_data_dir", default = os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument("--test_data_dir", default = os.environ['SM_CHANNEL_TEST'])
    parser.add_argument("--vocabulary_dir", default = os.environ['SM_CHANNEL_VOCABULARY'])
    parser.add_argument("--data_output_dir", default = os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument("--model_dir", default = os.environ['SM_MODEL_DIR'])
    args = parser.parse_args()
    return args

#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    args = parse_args()
    
    # Size of the embedding vector for each token
    embed_dim = int(args.embed_dim)
    # Size of the lstm output
    lstm_size = int(args.lstm_size)
    # Whether to run a bidirectional LSTM
    bidirectional = bool(args.bidirectional)
    # Number of LSTM layers
    num_layers = int(args.num_layers)
    # LSTM dropout
    dropout = float(args.dropout)
    # Learning rate for trianing the model
    learning_rate = float(args.learning_rate)
    # Number of epochs to run
    epochs = int(args.epochs)
    # Setting the threshold for positive and negative labels
    threshold = float(args.threshold)
    
    batch_size = int(args.batch_size)
    
    with open(args.model_dir + "/model_info.json", "w") as file:
        json.dump(vars(args), file) 
    
    report, model = train_model(args, embed_dim, lstm_size, bidirectional, num_layers, dropout, learning_rate, epochs, threshold, batch_size)
    













