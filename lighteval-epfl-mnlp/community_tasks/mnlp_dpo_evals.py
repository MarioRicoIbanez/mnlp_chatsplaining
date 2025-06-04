from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

def preference_pair(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        text_chosen=line["chosen"],
        text_rejected=line["rejected"],
        instruction="",
        choices = [],
        gold_index=0,
    )

task = LightevalTaskConfig(
    name="mnlp_dpo_evals",
    prompt_function=preference_pair,
    suite=["community"],
    hf_subset="",
    hf_repo="zechen-nlp/MNLP_dpo_demo",   # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.reward_model_acc],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=10 # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

# STORE YOUR EVALS
TASKS_TABLE = [task]
