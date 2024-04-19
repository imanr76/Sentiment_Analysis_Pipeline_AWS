import boto3
import json

def create_lambda_role(role_name, boto_session):
    """
    Creates a role that could be taken up by the AWS Lambda function that deploys the model. This role allows
    LAmbda to access the sagemaker model registry and to deploy the model to an endpoint. 
    
    
    Parameters
    ----------
    role_name : str
        name of the role that we are creating.
    boto_session : obj
        The boto3 session object to be used.

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
                            "Principal": {"Service": "lambda.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Role for Lambda to call SageMaker functions",
        )

        role_arn = response["Role"]["Arn"]

        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )

        response = iam.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess", RoleName=role_name
        )

        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using ARN from existing role: {role_name}")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]
