import boto3
import sagemaker
import os
from dotenv import load_dotenv
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig
from pipeline_steps.data_preparation_step import define_processing_step
from pipeline_steps.training_step import define_training_step
from pipeline_steps.evaluation_step import define_evaluation_step
from pipeline_steps.create_register_step import define_create_register_step
from pipeline_steps.deployment_step import define_deployment_step
from pipeline_steps.condition_step import define_condition_step
from datetime import datetime



def setup_sagemaker(local = True):
    """
    Sets up the sagemaker and boto3 sessions required for running the processing job.

    Parameters
    ----------
    local : boolean
        Whether the script is running locally or inside sagemaker notebooks.

    Returns
    -------
    role : str, obj
        ARN role for the sagemaker session.
    bucket : str
        The default bucket name for the sagemaker session.
    region : str
        The region of the sagemaker and boto3 session.
    boto3_session : obj
    sagemaker_Sess : obj
    """
    # IF running the script locally
    if local:
        load_dotenv()
        role = os.getenv("ROLE")
        profile_name  = os.getenv("AWS_PROFILE")
        boto3_session = boto3.session.Session(profile_name = profile_name)
        sagemaker_Sess = sagemaker.Session(boto_session = boto3_session)
    # If running the code from a sagemaker notebook
    else:
        boto3_session = boto3.session.Session()
        sagemaker_Sess = sagemaker.Session()
        role = sagemaker_Sess.get_execution_role()
    
    region = sagemaker_Sess.boto_region_name
    bucket = sagemaker_Sess.default_bucket()
    
    return role, bucket, region, boto3_session, sagemaker_Sess


def start_pipeline(pipeline_name = 'Pipeline-2024-04-16-20-41', max_len = 500, train_size = 0.8,
                   validation_size = 0.15, test_size = 0.05, processing_instacne_type = "ml.t3.large", 
                   processing_instance_count = 1, embed_dim = 20, lstm_size = 20, bidirectional = True,
                   num_layers = 1, dropout = 0, learning_rate = 0.001, epochs = 5, threshold = 0.5, 
                   batch_size = 32, train_instance_type = "ml.m5.xlarge", train_instance_count = 1, 
                   evaluation_instacne_type = "ml.t3.large", evaluation_instance_count = 1, 
                   deployment_instance_count = 1, deployment_instance_type = "ml.t2.medium",
                   min_accuracy_threshold = 50, cache = True, model_package_group_name = "group1"):
    """
    

    Parameters
    ----------
    pipeline_name : str, optional
        DESCRIPTION. The default is 'Pipeline-2024-04-16-20-41'.
    max_len : int, optional
        DESCRIPTION. The default is 500.
    train_size : float, optional
        DESCRIPTION. The default is 0.8.
    validation_size : float, optional
        DESCRIPTION. The default is 0.15.
    test_size : float, optional
        DESCRIPTION. The default is 0.05.
    processing_instacne_type : str, optional
        DESCRIPTION. The default is "ml.t3.large".
    processing_instance_count : int, optional
        DESCRIPTION. The default is 1.
    embed_dim : int, optional
        DESCRIPTION. The default is 20.
    lstm_size : int, optional
        DESCRIPTION. The default is 20.
    bidirectional : boolean, optional
        DESCRIPTION. The default is True.
    num_layers : int, optional
        DESCRIPTION. The default is 1.
    dropout : float, optional
        DESCRIPTION. The default is 0.
    learning_rate : float, optional
        DESCRIPTION. The default is 0.001.
    epochs : int, optional
        DESCRIPTION. The default is 5.
    threshold : float, optional
        DESCRIPTION. The default is 0.5.
    batch_size : int, optional
        DESCRIPTION. The default is 32.
    train_instance_type : str, optional
        DESCRIPTION. The default is "ml.m5.xlarge".
    train_instance_count : int, optional
        DESCRIPTION. The default is 1.
    evaluation_instacne_type : str, optional
        DESCRIPTION. The default is "ml.t3.large".
    evaluation_instance_count : int, optional
        DESCRIPTION. The default is 1.
    deployment_instance_count : int, optional
        DESCRIPTION. The default is 1.
    deployment_instance_type : str, optional
        DESCRIPTION. The default is "ml.t2.medium".
    min_accuracy_threshold : float, optional
        DESCRIPTION. The default is 50.
    cache : boolean, optional
        DESCRIPTION. The default is True.
    model_package_group_name : str, optional
        DESCRIPTION. The default is "group1".

    Returns
    -------
    pipeline_response : obj
        The ourput of the pipeline creation/update.
    pipeline_execution : obj
        The output of the pipeline run.
    endpoint_name : str
        The name of the deployed endpoint for inference.
    boto3_session : obj
        boto3 session object.
    pipeline_status : str
        The final status of the pipeline run.

    """
    
    # If caching is on, creates a cache configure object that caches the possible piepline steps. 
    if cache:
        cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    else:
        cache_config = CacheConfig(enable_caching=False, expire_after="30d")
    # creating the sagemaker and boto3 sessions and caching relavant information
    role, bucket, region, boto3_session, sagemaker_Sess = setup_sagemaker(True)
    # Creating a tupe of the infor to pas to the steps
    session_info = (role, bucket, region, boto3_session, sagemaker_Sess )
    # Defining the processing step
    # This step, processes the raw data and prepares the train, validation and test datasets for training
    # and evaluating the model
    processing_step = define_processing_step(session_info, processing_instacne_type, 
                                processing_instance_count, cache_config,\
                                max_len, train_size, validation_size, test_size)
    # Defining the training step  
    # Trains the model and publishes model artifcats and information about the training steps 
    # to S3
    training_step = define_training_step(session_info, processing_step, train_instance_type, train_instance_count,
                              cache_config, embed_dim, lstm_size, bidirectional, num_layers,
                              dropout, learning_rate, epochs, threshold, batch_size)
    # Defining the evaluation step     
    # Creates a property file that contains the accuracy of the model on the test data
    evaluation_step, evaluation_report = define_evaluation_step(session_info, processing_step, training_step, 
                               evaluation_instacne_type, evaluation_instance_count, cache_config)
    # Defining the register and create model step. If the evaluation criteria is met, 
    # the model package will be created and registered to Sagemaker model registry      
    register_step, create_step = define_create_register_step(session_info, evaluation_step, training_step,
                                    model_package_group_name)
    # Defining the deploy step. Deploys the model and creates an inference endpoint. 
    deploy_step, endpoint_name = define_deployment_step(session_info, create_step, pipeline_name, 
                               deployment_instance_count = deployment_instance_count,
                               deployment_instance_type = deployment_instance_type)
    # Defining the condition step. Checks the evaluation parameter calculated in the evaluation step.
    # If the calculated accuracy is higher then runs the create and register step and deployment step. 
    condition_step = define_condition_step(register_step, create_step, deploy_step,
                              evaluation_step, evaluation_report,
                              min_accuracy_threshold = min_accuracy_threshold)
    # Defining a pipeline object.
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[],
        steps=[processing_step, training_step, evaluation_step, condition_step],
        sagemaker_session=sagemaker_Sess
    )
    
    # Creating or updating the pipeline. If the pipeline name already exists, just updates it
    # and if caching is enabled, uses the caches steps if the parameters to that step are unchanged. 
    pipeline_response = pipeline.upsert(role_arn=role)
    # Executing the pipeline
    pipeline_execution = pipeline.start()
    
    print("Waiting for the pipeline run to finish...")
    # Waiting for the pipeline run to finish 
    pipeline_execution.wait(delay = 10, max_attempts = 240)
    
    execusion_info = pipeline_execution.describe()
    
    pipeline_status = execusion_info["PipelineExecutionStatus"]
    
    if pipeline_status == "Succeeded":
        print("Pipeline run was successful")
    
    else:
        print("Pipeline run failed")
    
    return pipeline_response, pipeline_execution, endpoint_name, boto3_session, pipeline_status


    
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
    
    pipeline_response, pipeline_execution, endpoint_name, boto3_session, pipeline_status = start_pipeline(pipeline_name, max_len, train_size ,
                       validation_size, test_size, processing_instacne_type, 
                       processing_instance_count, embed_dim, lstm_size, bidirectional,
                       num_layers, dropout, learning_rate, epochs, threshold, 
                       batch_size, train_instance_type, train_instance_count, 
                       evaluation_instacne_type , evaluation_instance_count, 
                       deployment_instance_count, deployment_instance_type,
                       min_accuracy_threshold, cache, model_package_group_name)
