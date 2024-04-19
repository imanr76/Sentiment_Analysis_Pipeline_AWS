import boto3
import json
import sagemaker
import os
from dotenv import load_dotenv
import subprocess
import time
from datetime import datetime

def create_eventbridge_role(role_name, boto_session, pipeline_arn):
    iam = boto_session.client("iam")
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "events.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Role for EventBridge to trigger Sagemaker Pipelines",
        )

        role_arn = response["Role"]["Arn"]

        response = iam.put_role_policy(
            RoleName=role_name,
            PolicyName="Amazon_EventBridge_Invoke_SageMaker_Pipeline1",
            PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:StartPipelineExecution"
                    ],
                    "Resource": [
                        pipeline_arn
                    ]
                }
            ]
        })
        )

        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using ARN from existing role: {role_name}")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]


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


def setup_eventbridge_trigger(pipeline_arn, session_info = None):

    
    if not session_info:
        role, bucket, region, boto3_session, sagemaker_Sess = setup_sagemaker(True)
    else:
        role, bucket, region, boto3_session, sagemaker_Sess = session_info
    
    
    response = create_eventbridge_role("eventbridege-role", boto3_session, pipeline_arn)
    
    pattern_file = "event_pattern.json"
    
    now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    rule_name = "sagemaker-pipeline-rule-" + now_time
    
    command1 = f"aws events put-rule\
                --name {rule_name}\
                --event-pattern file://{pattern_file}\
                --role-arn {response}\
                --profile {boto3_session.profile_name}"
    
    command2 = f"aws events put-targets \
                 --rule {rule_name} \
                 --event-bus-name default \
                 --targets '[{{\"Id\": \"first_param\", \"Arn\": \"{pipeline_arn}\", \"RoleArn\": \"{response}\"}}]'"
    
    
    process1 = subprocess.run(command1, capture_output=True, text=True, shell=True)
    
    print(process1.stdout)
    print(process1.stderr)
    # Get the command's return code
    return_code = process1.returncode
    print(f"Return Code: {return_code}\n")
    
    time.sleep(5)
        
    process2 = subprocess.run(command2, capture_output=True, text=True, shell=True)   
        
    print(process2.stdout)
    print(process2.stderr)
    # Get the command's return code
    return_code = process2.returncode
    print(f"Return Code: {return_code}\n") 
    
    
#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":    
    
    pipeline_arn = "arn:aws:sagemaker:ca-central-1:397567358266:pipeline/Pipeline-2024-04-16-20-41"
    
    setup_eventbridge_trigger(pipeline_arn)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    