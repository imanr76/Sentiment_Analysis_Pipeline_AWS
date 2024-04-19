from pipeline import start_pipeline
from datetime import datetime
from inference import make_inference
import time

def wait_for_endpoint(endpoint_name, boto_session):
    """
    Waits until the creation of the endpoint is complete and the endpoint becomes available for inference.

    Parameters
    ----------
    endpoint_name : str
        The name of the endpoint to observe.
    boto_session : obj
        boto3 session.

    Raises
    ------
    RuntimeError
        If endpoint is not successfully created this error is raised.

    Returns
    -------
    None.

    """
    
    sagemaker = boto_session.client("sagemaker")
    while True:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        if status == "InService":
            print("Endpoint is now available.")
            break
        elif status == "Failed":
            print("Endpoint creation failed.")
        print("Endpoint status:", status)
        time.sleep(10)
        
#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    # Getting the current time for naming purposes 
    now_time = datetime.now()
    # Name of the pipeline, if new, a new pipeline is created, otherwise the same pipeline is just updated. 
    pipeline_name = 'Pipeline-2024-04-16-20-41'
    # Maximum review text sequence length
    max_len = 500
    # Fraction of training data of all data
    train_size = 0.8
    # Fraction of validation data of all data
    validation_size = 0.15
    # Fraction of test data of all data
    test_size = 0.05
    # The type of instance to run the job on
    processing_instacne_type = "ml.t3.large"
    # The number of instances to use for processing job
    processing_instance_count = 1
    # Size of the embedding vector for each token
    embed_dim = 20
    # Size of the lstm output
    lstm_size = 20
    # Whether to run a bidirectional LSTM
    bidirectional = True
    # Number of LSTM layers
    num_layers = 1
    # LSTM dropout
    dropout = 0
    # Learning rate for trianing the model
    learning_rate = 0.001
    # Number of epochs to run
    epochs = 5
    # Setting the threshold for positive and negative labels
    threshold = 0.5
    # The batch size to be used during model training
    batch_size = 32
    # The instance type (VM type) for the training job 
    train_instance_type = "ml.m5.xlarge"
    # The number of VM nodes to use for training job
    train_instance_count = 1
    # The type of instance to run the evaluation job on
    evaluation_instacne_type = "ml.t3.large"
    # The number of VM nodes to use for evaluation job
    evaluation_instance_count = 1
    # The number of VM nodes to use for deployment and inference    
    deployment_instance_count = 1
    # The type of VM to use for deployment and inference    
    deployment_instance_type = "ml.t2.medium"
    #The minimum acceptable threshold for model accuracy to deploy the model
    min_accuracy_threshold = 50
    # Whether to cache steps of the pipeline
    cache = True
    # The name of the model package group, all the trained models will be saved as different versions of this model
    model_package_group_name = "group1"
    # Running the pipelne
    pipeline_response, pipeline_execution, endpoint_name, boto3_session, pipeline_status\
        = start_pipeline(pipeline_name, max_len, train_size ,
                       validation_size, test_size, processing_instacne_type, 
                       processing_instance_count, embed_dim, lstm_size, bidirectional,
                       num_layers, dropout, learning_rate, epochs, threshold, 
                       batch_size, train_instance_type, train_instance_count, 
                       evaluation_instacne_type , evaluation_instance_count, 
                       deployment_instance_count, deployment_instance_type,
                       min_accuracy_threshold, cache, model_package_group_name)
    # Waits for the pipeline run to finish and for the model endpoint to 
    # become responsive, then sends HTTP requests to the model and gets
    # a prediction
    if pipeline_status == "Succeeded":
        
        print("Waiting for endpoint to become avilable..\n.")
        
        review = "I absoloutley love this product, it is amazing, definitely recommend you to buy."
        
        wait_for_endpoint(endpoint_name, boto3_session)
        
        print("Getting some predictions\n")
        
        make_inference(review, endpoint_name)
        
    
    
    
    
    