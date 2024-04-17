import boto3
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.steps import CacheConfig


def define_processing_step(session_info, processing_instacne_type = "ml.t3.large", 
                           processing_instance_count = 1, cache_config = CacheConfig(enable_caching = False, expire_after = "30d"),\
                           max_len = 500, train_size = 0.8, validation_size = 0.15, test_size = 0.05):
    
    role, bucket, region, boto3_session, sagemaker_Sess = session_info
    
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
    
    # Defining the processing job 
    processor = SKLearnProcessor(framework_version = "0.23-1",role = role,\
                                 instance_type = processing_instacne_type,\
                                 instance_count = processing_instance_count,
                                 env = {'AWS_DEFAULT_REGION': region},
                                 sagemaker_session = sagemaker_Sess)

    
    processing_step = ProcessingStep(
        name='DataPreparation', 
        code='src/data_preparation.py',
        processor=processor,
        inputs=processing_inputs,
        outputs=processing_outputs,
        job_arguments=["--max-len", str(max_len),
                   "--train-size", str(train_size),
                   "--validation-size", str(validation_size),
                   "--test-size", str(test_size)],
        cache_config = cache_config)   
    
    return processing_step