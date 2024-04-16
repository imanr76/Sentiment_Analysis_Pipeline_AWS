import boto3
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
import os
from dotenv import load_dotenv
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.pytorch.estimator import PyTorch as PytorchEstimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.properties import PropertyFile


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
        Teh region of the sgaemaker and boto3 session.
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


role, bucket, region, boto3_session, sagemaker_Sess = setup_sagemaker(True)


cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")

processing_inputs = [ProcessingInput(input_name = "raw_input_data",\
                                      source = "s3://" + bucket + "/raw_data/womens_clothing_ecommerce_reviews.csv",\
                                      destination = '/opt/ml/processing/data/raw_data/')]
# Defining the processing job outputs.
processing_outputs = [ProcessingOutput(output_name = "training_data",\
                                       source = '/opt/ml/processing/data/training/',\
                                       destination = "s3://" + bucket + "/data/training/",\
                                       s3_upload_mode='EndOfJob'),
                      ProcessingOutput(output_name = "validation_data",\
                                       source = '/opt/ml/processing/data/validation/',\
                                       destination = "s3://" + bucket + "/data/validation/",\
                                       s3_upload_mode='EndOfJob'),\
                      ProcessingOutput(output_name = "test_data",\
                                       source = '/opt/ml/processing/data/test/',\
                                       destination = "s3://" + bucket + "/data/test/",\
                                       s3_upload_mode='EndOfJob'),
                      ProcessingOutput(output_name = "vocabulary",\
                                       source = '/opt/ml/processing/models/',\
                                       destination = "s3://" + bucket + "/models/vocabulary",\
                                       s3_upload_mode='EndOfJob')]
    
# The type of instance to run the job on
processing_instacne_type = "ml.t3.large"
# The number of instances to use for processing job
processing_instance_count = 1

# Defining the processing job 
processor = SKLearnProcessor(framework_version = "0.23-1",role = role,\
                             instance_type = processing_instacne_type,\
                             instance_count =1,
                             env={'AWS_DEFAULT_REGION': region},
                             sagemaker_session = sagemaker_Sess)


    
# Maximum review text sequence length
max_len = 500
# Fraction of training data of all data
train_size = 0.8
# Fraction of validation data of all data
validation_size = 0.15
# Fraction of test data of all data
test_size = 0.05

processing_step = ProcessingStep(
    name='Processing', 
    code='src/data_preparation.py',
    processor=processor,
    inputs=processing_inputs,
    outputs=processing_outputs,
    job_arguments=["--max-len", str(max_len),
               "--train-size", str(train_size),
               "--validation-size", str(validation_size),
               "--test-size", str(test_size)],
    cache_config = cache_config)   




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

batch_size = 32

train_instance_type = "ml.m5.xlarge"

train_instance_count = 1

hyperparameters = {
                    "embed_dim" : embed_dim,
                    "lstm_size" : lstm_size,
                    "bidirectional" : bidirectional,
                    "num_layers" : num_layers,
                    "dropout" : dropout,
                    "learning_rate" : learning_rate,
                    "epochs" : epochs,
                    "threshold" : threshold,
                    "batch_size" : batch_size
                    }


data_channels = {"train" : TrainingInput(s3_data=processing_step.properties.ProcessingOutputConfig\
                                         .Outputs['training_data'].S3Output.S3Uri),
                 "validation" : TrainingInput(s3_data=processing_step.properties.ProcessingOutputConfig\
                                                          .Outputs['validation_data'].S3Output.S3Uri),
                 "test" : TrainingInput(s3_data=processing_step.properties.ProcessingOutputConfig\
                                                          .Outputs['test_data'].S3Output.S3Uri),
                 "vocabulary" : TrainingInput(s3_data=processing_step.properties.ProcessingOutputConfig\
                                                          .Outputs['vocabulary'].S3Output.S3Uri)}
estimator = PytorchEstimator(
                            entry_point = "./src/training.py",
                            framework_version = "1.13",
                            py_version = "py39",
                            role = role,
                            instance_count = train_instance_count,
                            instance_type = train_instance_type,
                            hyperparameters = hyperparameters,
                            input_mode = 'File',
                            output_path = "s3://" + bucket + "/models/",
                            sagemaker_session = sagemaker_Sess
                            )


training_step = TrainingStep(
    name='Train',
    estimator=estimator,
    inputs=data_channels,
    cache_config=cache_config
)




evaluation_report = PropertyFile(
    name='EvaluationReport',
    output_name='metrics',
    path='evaluation.json'
)

processing_inputs = [ProcessingInput(input_name = "test_data",\
                                      source = processing_step.properties.ProcessingOutputConfig.Outputs['test_data'].S3Output.S3Uri,\
                                      destination = '/opt/ml/processing/data/'),
                     ProcessingInput(input_name = "model_artifacts",\
                                      source = training_step.properties.ModelArtifacts.S3ModelArtifacts,\
                                      destination = '/opt/ml/processing/model/')]
# Defining the processing job outputs.
processing_outputs = [ProcessingOutput(output_name = "metrics",\
                                       source = '/opt/ml/processing/output/',\
                                       s3_upload_mode='EndOfJob')]
    
# The type of instance to run the job on
processing_instacne_type = "ml.t3.large"
# The number of instances to use for processing job
processing_instance_count = 1

# Defining the processing job 
processor = SKLearnProcessor(framework_version = "0.23-1",role = role,\
                             instance_type = processing_instacne_type,\
                             instance_count =1,
                             env={'AWS_DEFAULT_REGION': region},
                             sagemaker_session = sagemaker_Sess)

    
evaluation_step = ProcessingStep(
    name = 'Evaluation', 
    code = 'src/evaluation.py',
    processor = processor,
    inputs = processing_inputs,
    outputs = processing_outputs,
    job_arguments = ["--max-len", str(max_len)],
    property_files = [evaluation_report],
    cache_config=cache_config)   


pipeline = Pipeline(
    name='Processing-Training-Pipeline4',
    parameters=[],
    steps=[processing_step, training_step, evaluation_step],
    sagemaker_session=sagemaker_Sess
)

response = pipeline.create(role_arn=role)
execution = pipeline.start()





