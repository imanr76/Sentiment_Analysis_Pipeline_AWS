# Custom Lambda Step
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)

from pipeline_steps.src import lambda_role_creator
from datetime import datetime
import time
now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def define_deployment_step(session_info, create_step, pipeline_name, 
                           deployment_instance_count = 1, deployment_instance_type = "ml.t2.medium"):
    """
    Creates a lambda step which runs a lambda function. The lambda function deploys the created model
    to an endpoint. 

    Parameters
    ----------
    session_info : obj
        Information about the boto3 and sagemaker sessions.
    create_step : obj
        The output of the create step.
    pipeline_name : str
        The name of the pipeline.
    deployment_instance_count : int, optional
        The number of VM nodes to use for deployment and inference . The default is 1.
    deployment_instance_type : str, optional
        The type of VM to use for deployment and inference . The default is "ml.t2.medium".

    Returns
    -------
    deploy_step : obj
        The output of the deployment step.
    endpoint_name : str
        The name of the created endpoint.

    """
    
    role, bucket, region, boto3_session, sagemaker_Sess = session_info
    
    
    
    function_name = pipeline_name + "-lambda-function-" + now_time
    endpoint_config_name = pipeline_name + "-endpoint-config-"
    endpoint_name = pipeline_name + "-endpoint-"
    
    lambda_role = lambda_role_creator.create_lambda_role(pipeline_name + "-deployment-role", boto3_session)
    
    # Sometimes, it takes time for the created role to become available which causes errors.
    # The following run waits for some time to  make sure that the role is avilable.
    time.sleep(5)
    
    func = Lambda(
        function_name=function_name,
        execution_role_arn=lambda_role,
        script="pipeline_steps/src/lambda_function.py",
        handler="lambda_function.lambda_handler",
    )
    
    output_param_1 = LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String)
    output_param_2 = LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String)
    
    deploy_step = LambdaStep(
        name = "Deploy",
        lambda_func = func,
        inputs = {
            "model_name": create_step.properties.ModelName,
            "endpoint_config_name": endpoint_config_name,
            "endpoint_name": endpoint_name,
            "deployment_instance_count": deployment_instance_count,
            "deployment_instance_type": deployment_instance_type
        },
        outputs = [output_param_1, output_param_2],
    )
    
    return deploy_step, endpoint_name