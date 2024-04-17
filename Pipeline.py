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


cache_config = CacheConfig(enable_caching=True, expire_after="30d")

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



from sagemaker.workflow.model_step import ModelStep
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.workflow.pipeline_context import PipelineSession

pipeline_session = PipelineSession(boto_session = boto3_session)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json"
    )
)

pytorch_model = PyTorchModel(
   model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
   entry_point='deployment.py',
   source_dir='src/',
   framework_version = "1.13",
   py_version = "py39",
   role=role,
   sagemaker_session = pipeline_session)


register_step_arguments = pytorch_model.register(
   content_types=["application/json"],
   response_types=["application/json"],
   inference_instances=["ml.t2.medium"],
   transform_instances=["ml.m5.large"],
   model_package_group_name='group1',
   model_metrics=model_metrics,
   approval_status="Approved"
)

register_step = ModelStep(
   name = "ModelRegister",
   step_args = register_step_arguments,

)

create_step_arguments = pytorch_model.create("ml.t2.medium")

create_step = ModelStep(
    name="ModelCreate",
    step_args=create_step_arguments,
)


register_properties = register_step.properties


from src import iam_helper

from datetime import datetime


lambda_role = iam_helper.create_lambda_role("lambda-deployment-role", boto3_session)

now_time = datetime.now()
# Custom Lambda Step
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)

function_name = "sagemaker-lambda-step-endpoint-deploy-" + now_time.strftime("%Y-%m-%d-%H-%M")

# Lambda helper class can be used to create the Lambda function
func = Lambda(
    function_name=function_name,
    execution_role_arn=lambda_role,
    script="src/lambda_helper.py",
    handler="lambda_helper.lambda_handler",
)

endpoint_config_name = "demo-lambda-deploy-endpoint-config-" + now_time.strftime("%Y-%m-%d-%H-%M")
endpoint_name = "demo-lambda-deploy-endpoint-" + now_time.strftime("%Y-%m-%d-%H-%M")

output_param_1 = LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String)
output_param_2 = LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String)
output_param_3 = LambdaOutput(output_name="other_key", output_type=LambdaOutputTypeEnum.String)

deploy_step = LambdaStep(
    name="LambdaStep",
    lambda_func=func,
    inputs={
        "model_name": create_step.properties.ModelName,
        "endpoint_config_name": endpoint_config_name,
        "endpoint_name": endpoint_name,
    },
    outputs=[output_param_1, output_param_2, output_param_3],
)


from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)

minimum_accuracy_condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step=evaluation_step,
        property_file=evaluation_report,
        json_path="metrics.accuracy.value",
    ),
    right=90 # minimum accuracy threshold
)

condition_step = ConditionStep(
    name="AccuracyCondition",
    conditions=[minimum_accuracy_condition],
    if_steps=[register_step, create_step, deploy_step], # successfully exceeded or equaled the minimum accuracy, continue with model registration
    else_steps=[], # did not exceed the minimum accuracy, the model will not be registered
)





pipeline = Pipeline(
    name='Processing-Training-Pipeline-2024-04-16-20-41',
    parameters=[],
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=sagemaker_Sess
)

response = pipeline.update(role_arn=role)
execution = pipeline.start()





