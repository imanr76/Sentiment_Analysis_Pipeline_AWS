from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)


def define_condition_step(register_step, create_step, deploy_step,
                          evaluation_step, evaluation_report,
                          min_accuracy_threshold = 50):
    """
    Checks whether the trained model meets the accuracy threshold. 
    If it passes, model is then registered, created and deployed to an endpoint for real time inference.
    If it fails, the pipeline stops. 

    Parameters
    ----------
    register_step : obj
        The output of the register step.
    create_step : obj
        The output of the create step.
    deploy_step : obj
        The output of the deployment step.
    evaluation_step : obj
        The output of the evaluation step.
    evaluation_report : obj
        The evaluation property file generated in the evaluation step.
    min_accuracy_threshold : float, optional
        The minimu accuracy criteria for continuing or stopping the pipeline. The default is 50.

    Returns
    -------
    condition_step : obj
        The output of the condtion step.

    """
    # Defining the condition 
    minimum_accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=evaluation_step,
            property_file=evaluation_report,
            json_path="metrics.accuracy.value",
        ),
        right=min_accuracy_threshold # minimum accuracy threshold
    )
    # Defining the condition step
    condition_step = ConditionStep(
        name="Condition",
        conditions=[minimum_accuracy_condition],
        if_steps=[register_step, create_step, deploy_step], # successfully exceeded or equaled the minimum accuracy, continue with model registration
        else_steps=[], # did not exceed the minimum accuracy, the model will not be registered
    )
    
    return condition_step