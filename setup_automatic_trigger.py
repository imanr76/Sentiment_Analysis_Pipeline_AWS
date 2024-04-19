import boto3
import json
import sagemaker
import os
from dotenv import load_dotenv
import subprocess
import time
from datetime import datetime

def create_eventbridge_role(role_name, boto_session, pipeline_arn):
    """
    Creates a role that could be taken up by the AWS EventBridge Rule. This role allows
    EventBridge to run the Sagemaker pipeline. 
    
    NOTE: To change the event that triggers the pipeline change the contents of the event_pattern.json file
    
    Parameters
    ----------
    role_name : str
        name of the role that we are creating.
    boto_session : obj
        The boto3 session object to be used.
    pipeline_arn : str
        The AWS ARN of the pipeline that this role could trigger .

    Returns
    -------
    role_arn : str
        The AWS ARN of the role just created.

    """
    # Instansiating a boto3 IAM client
    iam = boto_session.client("iam")
    # Trying to create the role and attach the required policies
    # If role already exists, uses the preexisting role.
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
    """
    Creates an EventBridge Rule. This rule uses the default event bus which observes different 
    information that are sent to it. Whenver the information match the JSON  saved in the event_pattern.json file, 
    it triggers the target. 
    This specific rule, receives a notification from S3
    whenever a file is uploaded into a specific S3 location and it triggers the pipeline 
    and runs it. 

    Parameters
    ----------
    pipeline_arn : str
        The AWS ARN of the pipeline that we would like to trigger.
    session_info : tupe(str, obj), optional
        boto3 and sagemaker session information. The default is None.

    Returns
    -------
    None.

    """
    # If session info is not given, createit. 
    if not session_info:
        role, bucket, region, boto3_session, sagemaker_Sess = setup_sagemaker(True)
    else:
        role, bucket, region, boto3_session, sagemaker_Sess = session_info
    
    # Creating the required role that the EventBridge Rule should take
    response = create_eventbridge_role("eventbridege-role", boto3_session, pipeline_arn)
    # The location of event pattern file to match
    pattern_file = "event_pattern.json"
    
    now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    rule_name = "sagemaker-pipeline-rule-" + now_time
    # The command for creating the EventBridge Rule
    command1 = f"aws events put-rule\
                --name {rule_name}\
                --event-pattern file://{pattern_file}\
                --role-arn {response}\
                --profile {boto3_session.profile_name}"
    # The command for assigning Sagemaker pipeline as the Rule target.
    command2 = f"aws events put-targets \
                 --rule {rule_name} \
                 --event-bus-name default \
                 --targets '[{{\"Id\": \"first_param\", \"Arn\": \"{pipeline_arn}\", \"RoleArn\": \"{response}\"}}]'"
    
    # Running the commands using the subprocess module. 
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
    # Setting up an EventBridge Rule for the pipeline_arn pipeline. 
    setup_eventbridge_trigger(pipeline_arn)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    