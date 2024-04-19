import json
import boto3
from datetime import datetime

def lambda_handler(event, context):
    """
    Lambda function handler that deploy the registered and created model. 

    Parameters
    ----------
    event : dict
        Event information passed to Lambda.
    context : dict
        Context of the Lambda function.

    Returns
    -------
    dict
        The response dictionary containing the status code and a message.

    """
    
    now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    sm_client = boto3.client("sagemaker")

    # The name of the model created in the Pipeline CreateModelStep
    model_name = event["model_name"]
    
    # Adding the time and date to the endpoint_name and endpoint_config_ name to make them unique everytime the function is run. 
    # If they already exist, it raises error
    endpoint_config_name = event["endpoint_config_name"] + now_time 
    endpoint_name = event["endpoint_name"] + now_time

    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": event["deployment_instance_type"],
                "InitialVariantWeight": 1,
                "InitialInstanceCount": event["deployment_instance_count"],
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )

    return {
        "statusCode": 200,
        "body": json.dumps("Created Endpoint!"),
    }