import boto3
import sagemaker
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.steps import CacheConfig


def define_evaluation_step(session_info, processing_step, training_step, 
                           evaluation_instacne_type = "ml.t3.large", evaluation_instance_count = 1,
                           cache_config = CacheConfig(enable_caching = False, expire_after = "30d")):
    
    role, bucket, region, boto3_session, sagemaker_Sess = session_info
    
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
        

    
    # Defining the processing job 
    processor = SKLearnProcessor(framework_version = "0.23-1",role = role,\
                                 instance_type = evaluation_instacne_type,\
                                 instance_count = evaluation_instance_count,
                                 env={'AWS_DEFAULT_REGION': region},
                                 sagemaker_session = sagemaker_Sess)
    
        
    evaluation_step = ProcessingStep(
        name = 'Evaluation', 
        code = 'pipeline_steps/src/evaluation.py',
        processor = processor,
        inputs = processing_inputs,
        outputs = processing_outputs,
        job_arguments = ["--max-len", str(1)],
        property_files = [evaluation_report],
        cache_config = cache_config)   
    
    return evaluation_step, evaluation_report