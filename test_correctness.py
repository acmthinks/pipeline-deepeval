from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5
)

test_cases = [
    LLMTestCase(
        input="What is the capital of France?",
        # Replace this with the actual output from your LLM application
        actual_output="The capital of France is Paris.",
        expected_output="Paris is the capital of France. Paris is located along the Seine River.",
        retrieval_context=["France is a country in Western Europe."]
    )
]

evaluate(test_cases, [correctness_metric])
