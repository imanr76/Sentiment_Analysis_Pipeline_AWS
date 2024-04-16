# Importing the required libraries

import os
import argparse
os.system('pip install nltk')
os.system('pip install torch==1.13.0')
os.system('pip install torchtext==0.14.0')
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import re
from torchtext.vocab import vocab
from torch.utils.data import Dataset
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download('wordnet')


#------------------------------------------------------------------------------
# Function Definitions
def parse_args():
    parser = argparse.ArgumentParser(description='processing job')
    
    parser.add_argument("--max-len", default = 500)
    parser.add_argument("--train-size", default = 0.8)
    parser.add_argument("--validation-size", default = 0.15)
    parser.add_argument("--test-size", default = 0.05)
    args = parser.parse_args()
    return args

def rating_to_sentiment(rating):
    """
    Parameters
    ----------
    rating : int
        The star rating of the item from 1 to 5.

    Returns
    -------
    sentiment : int
        Implied sentiment of the star rating, assumes ratings between 1 and 3 (inclusive) to be
        negative (0) and rating more than 3 to be positive (1).
    """
    
    if rating in {1, 2, 3}:
        return 0
    else:
        return 1

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

def create_vocab(text, tokenizer, lemmatizer, unk_token, pad_token):
    """
    Creates a vocabulary based on the input text corpus that assigns an index 
    to each token.

    Parameters
    ----------
    text : str
        The text corpus used for token extraction.
    tokenizer : obj
        Tokenizer object for tokenization of the text.
    lemmatizer : obj
        Lemmantizer onject for Llmmantization of the text.
    unk_token : str
        The symbol used for out of vocabulary tokens.
    pad_token : str
        The symbol used for displaying the padded indices of the sentences.

    Returns
    -------
    vocabulary : TYPE
        DESCRIPTION.

    """
    
    # Tookenizing the text
    tokenized_text = tokenizer(text)
    # Lemmantizing the text
    lemmatized_text = [lemmatizer.lemmatize(word) for word in tokenized_text]
    # Creating a hash map counting the instances of each token
    token_freqs = Counter(lemmatized_text)
    # Creating a vocabulary 
    vocabulary = vocab(token_freqs, min_freq = 10, specials = [pad_token, unk_token])
    # Setting the index that should be assigned to OOV tokens.
    vocabulary.set_default_index(1)
    return vocabulary

def process_reviews(review, tokenizer, lemmatizer, vocabulary, max_len, pad = True):
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

    if pad and len(review_processed) < max_len:
        review_processed.extend([0] * (max_len - len(review_processed)))
    elif len(review_processed) > max_len:
        review_processed = review_processed[:max_len]
    return review_processed

def convert_to_tensor(dataframe):
    """
    Converts the dataframe values into a list of tensors and appending the sentiment for each
    review to the review tensor.

    Parameters
    ----------
    dataframe : pandas DataFrame
        Dataframe whose data to be converted.

    Returns
    -------
    combined_tensor : list[torch.tensor]
        A list of torch tensors containing the indices of each review and the sentiment 
        as the last element.
    """
    
    # Converting the dataset values to lists
    review_processed_values = dataframe['review_processed'].tolist()
    sentiment_values = dataframe['sentiment'].tolist()
    #Converting dataset values to tensors
    review_processed_tensor = torch.tensor(review_processed_values)
    sentiment_tensor = torch.tensor(sentiment_values)
    # Appending the sentiment to the review indices tensor as the last element
    sentiment_tensor = sentiment_tensor.unsqueeze(1)
    combined_tensor = torch.cat((review_processed_tensor, sentiment_tensor), dim=1)
    return combined_tensor
    

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

def process_data(max_len = 500, train_size = 0.8, validation_size = 0.15, test_size = 0.05):
    """
    Downloads the data from the S# bucket on AWS, transforms it, balances it, creates a vocbulary from the
    review texts, converts the text into sequences of indices, divides the data into 
    training, validation and test sets and save them as PyTorch datsets. 

    Parameters
    ----------
    max_len : int, optional
        Maximum review text sequence length. The default is 500.
    train_size : int, optional
        Fraction of training data of all data. The default is 0.8.
    validation_size : int, optional
        Fraction of validation data of all data. The default is 0.15.
    test_size : int, optional
        Fraction of test data of all data. The default is 0.05.

    Returns
    -------
    None.

    """
    #------------------------------------------------------------------------------
    # Reading and transforming the dataset
    
    # Creating the necessary directories and downloading the data

    print("\nReading the data\n")
    # Reading the data
    data = pd.read_csv("/opt/ml/processing/data/raw_data/womens_clothing_ecommerce_reviews.csv")
    # Keeping the useful columns
    data_transformed =  data[["Review Text", "Rating", "Class Name"]].copy()
    # Renaming the columns for convenience
    data_transformed.rename(columns = {"Review Text":'review', "Rating":"rating", "Class Name":"product_category"}, inplace = True)
    # dropping the rows wth empty cells 
    data_transformed.dropna(inplace = True)
    # Removing the data for product categories with less than 10 reviews
    data_transformed  = data_transformed.groupby("product_category").filter(lambda review: len(review) > 10)
    # Converting the star rating to sentiment and dropping the rating column as it is not needed anymore
    data_transformed["sentiment"] = data_transformed["rating"].apply(lambda rating: rating_to_sentiment(rating))
    data_transformed.drop(columns = "rating", inplace = True)
    # Saving the transformed dataset
    # data_transformed.to_csv("./data/raw_data/womens_clothing_ecommerce_reviews_transformed.csv", index = False)
    
    
    #------------------------------------------------------------------------------
    # Balancing the dataset
    print("\nBalancing the dataset\n")
    # Balancing the dataset based on the sentiments so we have the same number of reviews for both sentiments
    data_transformed_grouped_for_balance = data_transformed.groupby(["sentiment"])[["review","sentiment", "product_category"]]
    data_transformed_balanced = data_transformed_grouped_for_balance.apply(lambda x: \
                                    x.sample(data_transformed.groupby(["sentiment"]).size().min()))\
                                    .reset_index(drop = True)# Saving the balanced dataset
    # Saving the balanced dataset
    # data_transformed_balanced.to_csv("./data/raw_data/womens_clothing_ecommerce_reviews_balanced.csv", index = False)
    
    # Dividing the data into train, validation and test sets
    training_data, temp_data = train_test_split(data_transformed_balanced, test_size = 1 - train_size, random_state = 10)
    validation_data, test_data = train_test_split(temp_data, test_size = test_size / (test_size + validation_size), random_state = 10)
    
    # Saving the train, validation and test datasets
    # training_data.to_csv("./data/training/womens_clothing_ecommerce_reviews_balanced_training.csv", index = False)
    # validation_data.to_csv("./data/validation/womens_clothing_ecommerce_reviews_balanced_validation", index = False)
    # test_data.to_csv("./data/test/womens_clothing_ecommerce_reviews_balanced_test", index = False)
    
    #------------------------------------------------------------------------------
    # Preprocessing the data for the NLP task
    print("\nApplying feature engineering tasks\n")
    # Creating a text corpus from the training and validation data
    # corpus_data = pd.concat([training_data["review"], validation_data["review"]], axis = 0)
    corpus = '\n'.join(training_data["review"].values)
    
    # Saing the text corpus for future references and use
    # with open("./data/corpus.txt", "w") as file:
    #     file.write(corpus)
    
    # Cleaning he corpus text
    corpus_cleaned = cleanup_text(corpus)
    # Creating a vocabulary from the text corpus
    vocabulary = create_vocab(corpus_cleaned, word_tokenize, WordNetLemmatizer(), "<unk>", "<pad>")
    # Saving the vocabulary for future reference and use
    torch.save(vocabulary, '/opt/ml/processing/models/vocabulary.pth')
    
    # Processing the reviews in the datasets and converting the review text to list of indices
    training_data["review_processed"] = training_data["review"]\
        .apply(lambda x: process_reviews(x, word_tokenize, WordNetLemmatizer(), vocabulary, max_len))
    validation_data["review_processed"] = validation_data["review"]\
        .apply(lambda x: process_reviews(x, word_tokenize, WordNetLemmatizer(), vocabulary, max_len))
    test_data["review_processed"] = test_data["review"]\
        .apply(lambda x: process_reviews(x, word_tokenize, WordNetLemmatizer(), vocabulary, max_len))
    
    # Keeping only the required columns of the datsets
    training_data_processed = training_data[["review_processed", "sentiment"]]
    validation_data_processed  = validation_data[["review_processed", "sentiment"]]
    test_data_processed  = test_data[["review_processed", "sentiment"]]
    
    # Saving the datasets for future use and reference
    # training_data_processed.to_csv("./data/training/training_data_processed.csv", index = False)
    # validation_data_processed.to_csv("./data/validation/validation_data_processed.csv", index = False)
    # test_data_processed.to_csv("./data/test/test_data_processed.csv", index = False)
    
    # Converting the dataframe data into tensors
    training_data_tensor = convert_to_tensor(training_data_processed)
    validation_data_tensor = convert_to_tensor(validation_data_processed)
    test_data_tensor = convert_to_tensor(test_data_processed)
    
    # Creating torch Datasets
    train_dataset = dataset(training_data_tensor)
    validaton_dataset = dataset(validation_data_tensor)
    test_dataset = dataset(test_data_tensor)
    print("\nSaving the datasets\n")
    # Saving the torch Datasets for future use and reference
    torch.save(train_dataset, "/opt/ml/processing/data/training/training_dataset.pth")
    torch.save(validaton_dataset, "/opt/ml/processing/data/validation/validation_dataset.pth")
    torch.save(test_dataset, "/opt/ml/processing/data/test/test_dataset.pth")
    
    print("\nProcessing complete\n")


#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    args = parse_args()
    
    # Maximum review text sequence length
    max_len = int(args.max_len)
    # Fraction of training data of all data
    train_size = float(args.train_size)
    # Fraction of validation data of all data
    validation_size = float(args.validation_size)
    # Fraction of test data of all data
    test_size = float(args.test_size)
    
    # Preprocessing the data
    process_data(max_len,  train_size, validation_size, test_size)
