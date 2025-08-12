from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

# Assuming DeepEval is configured to use your Granite model endpoint

test_cases = [
    LLMTestCase(
        input="What is the capital of France?",
        context=["France is a country in Western Europe."],
        actual_output="The capital of France is Paris.",
        expected_output="Paris is the capital of France. Paris is located along the Seine River.",
    ),
    LLMTestCase(
        input="How old is Ozzy Osbourne?",
        context=["Ozzy Osbourne is a British born singer and songwriter. He is most notable for the leading the band Black Sabbath and his solo singing career."],
        actual_output="Ozzy Osbourne, born John Michael Osbourne on December 3, 1948, is 73 years old, as of the knowledge cutoff in April 2024. He is an English singer",
        expected_output="Ozzy Osbourne passed away on July 22, 2025 (age 76 years)",
    )
]

# Define evaluation metrics
metrics = [
    HallucinationMetric(threshold=0.5)
]

# Run the evaluation
evaluate(test_cases, metrics)
