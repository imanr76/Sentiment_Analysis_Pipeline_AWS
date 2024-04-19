import json
import boto3
from datetime import datetime

def lambda_handler(event, context):
    
    now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    sm_client = boto3.client("sagemaker")

    # The name of the model created in the Pipeline CreateModelStep
    model_name = event["model_name"]

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