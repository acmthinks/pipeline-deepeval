from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

# Assuming DeepEval is configured to use your Granite model endpoint

test_cases = [
    LLMTestCase(
        input="What is the capital of France?",
        context=["France is a country in Western Europe."],
        #role="user",
        #content="What is the capital of France?",
        actual_output="Paris is the capital of France.",
        expected_output="Paris is the capital of France. France is a country in Western Europe that is located on the Seine River.",
    )
]

# Define evaluation metrics
metrics = [
    HallucinationMetric(threshold=0.5)
]

# Run the evaluation
evaluate(test_cases, metrics)
