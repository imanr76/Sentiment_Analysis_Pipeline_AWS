from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)


def define_condition_step(register_step, create_step, deploy_step,
                          evaluation_step, evaluation_report,
                          min_accuracy_threshold = 50):
    
    
    minimum_accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step=evaluation_step,
            property_file=evaluation_report,
            json_path="metrics.accuracy.value",
        ),
        right=min_accuracy_threshold # minimum accuracy threshold
    )
    
    condition_step = ConditionStep(
        name="Condition",
        conditions=[minimum_accuracy_condition],
        if_steps=[register_step, create_step, deploy_step], # successfully exceeded or equaled the minimum accuracy, continue with model registration
        else_steps=[], # did not exceed the minimum accuracy, the model will not be registered
    )
    
    return condition_step