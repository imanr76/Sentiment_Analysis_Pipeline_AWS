import sagemaker
import boto3
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.pytorch.estimator import PyTorch as PytorchEstimator
from sagemaker.workflow.steps import CacheConfig

def define_training_step(session_info, processing_step, train_instance_type = "ml.m5.xlarge", train_instance_count = 1,\
                         cache_config = CacheConfig(enable_caching = False, expire_after = "30d"),\
                         embed_dim = 20, lstm_size = 20, bidirectional = True, num_layers = 1,
                         dropout = 0, learning_rate = 0.001, epochs = 5, threshold = 0.5,
                         batch_size = 32):

    
    role, bucket, region, boto3_session, sagemaker_Sess = session_info
    
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
    
    processing_properties = processing_step.properties
    
    data_channels = {"train" : TrainingInput(s3_data=processing_properties.ProcessingOutputConfig\
                                             .Outputs['training_data'].S3Output.S3Uri),
                     "validation" : TrainingInput(s3_data=processing_properties.ProcessingOutputConfig\
                                                              .Outputs['validation_data'].S3Output.S3Uri),
                     "test" : TrainingInput(s3_data=processing_properties.ProcessingOutputConfig\
                                                              .Outputs['test_data'].S3Output.S3Uri),
                     "vocabulary" : TrainingInput(s3_data=processing_properties.ProcessingOutputConfig\
                                                              .Outputs['vocabulary'].S3Output.S3Uri)}
    estimator = PytorchEstimator(
                                entry_point = "pipeline_steps/src/training.py",
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
        name = 'Train',
        estimator = estimator,
        inputs = data_channels,
        cache_config = cache_config
    )
    
    return training_step