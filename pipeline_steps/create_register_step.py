from sagemaker.workflow.model_step import ModelStep
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.workflow.pipeline_context import PipelineSession


def define_create_register_step(session_info, evaluation_step, training_step,
                                model_package_group_name = "group1"):
    """
    Creates and registers the trained model in the training step to the model registry. 
    Passes the created model object for model deployment. 

    Parameters
    ----------
    session_info : obj
        Information about the boto3 and sagemaker sessions.
    evaluation_step : obj
        Output of the evaluation step.
    training_step : obj
        Output of the training step.
    model_package_group_name : obj, optional
        The name of the model package to which the model will be registered. The default is "group1".

    Returns
    -------
    register_step : obj
        Output of the registr step.
    create_step : TYPE
        Output of the create model step..

    """
    role, bucket, region, boto3_session, sagemaker_Sess = session_info
    
    pipeline_session = PipelineSession(boto_session = boto3_session)
    
    model_metrics = ModelMetrics(
        model_statistics = MetricsSource(
            s3_uri = "{}/evaluation.json".format(
                evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    # Creating amodel deployment object for future use
    pytorch_model = PyTorchModel(
       model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
       entry_point='deployment.py',
       source_dir='pipeline_steps/src/',
       framework_version = "1.13",
       py_version = "py39",
       role=role,
       sagemaker_session = pipeline_session)

    register_step_arguments = pytorch_model.register(
       content_types = ["application/json"],
       response_types = ["application/json"],
       inference_instances = ["ml.t2.medium"],
       transform_instances = ["ml.m5.large"],
       model_package_group_name = model_package_group_name,
       model_metrics = model_metrics,
       approval_status="Approved"
    )
    
    register_step = ModelStep(
       name = "RegisterModel",
       step_args = register_step_arguments,
    
    )
    
    create_step_arguments = pytorch_model.create("ml.t2.medium")
    
    create_step = ModelStep(
        name="CreateModel",
        step_args=create_step_arguments,
    )
    
    
    
    return register_step, create_step