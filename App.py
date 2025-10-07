"""
DeepEval Comprehensive Metrics Testing Suite
============================================
This script demonstrates how to use all major DeepEval metrics with sample data.

Prerequisites:
pip install deepeval
export OPENAI_API_KEY="your-api-key-here"


"""
import os
from dotenv import load_dotenv
load_dotenv()
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    HallucinationMetric,
    GEval
)

# ============================================
# RAG METRICS
# ============================================

def test_answer_relevancy():
    """Tests if the LLM output is relevant and concise to the input."""
    print("\n" + "="*50)
    print("Testing Answer Relevancy Metric")
    print("="*50)
    
    test_case = LLMTestCase(
        input="What are the health benefits of regular exercise?",
        actual_output="Regular exercise improves cardiovascular health, strengthens muscles, "
                     "enhances mental well-being, and helps maintain healthy weight. "
                     "It also reduces the risk of chronic diseases like diabetes and heart disease.",
        retrieval_context=[
            "Exercise has numerous health benefits including improved heart health and muscle strength.",
            "Physical activity is linked to better mental health and reduced anxiety.",
            "Regular exercise helps prevent type 2 diabetes and cardiovascular diseases."
        ]
    )
    
    metric = AnswerRelevancyMetric(threshold=0.7)
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


def test_faithfulness():
    """Tests if the LLM output is factually aligned with retrieval context."""
    print("\n" + "="*50)
    print("Testing Faithfulness Metric")
    print("="*50)
    
    test_case = LLMTestCase(
        input="When was the Eiffel Tower built?",
        actual_output="The Eiffel Tower was constructed between 1887 and 1889. "
                     "It was built for the 1889 World's Fair in Paris and stands 324 meters tall.",
        retrieval_context=[
            "The Eiffel Tower construction began in 1887 and was completed in 1889.",
            "The tower was built for the 1889 Exposition Universelle (World's Fair).",
            "The Eiffel Tower's height is 324 meters including antennas."
        ]
    )
    
    metric = FaithfulnessMetric(threshold=0.7)
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


def test_contextual_recall():
    """Tests if the retrieval context contains information from expected output."""
    print("\n" + "="*50)
    print("Testing Contextual Recall Metric")
    print("="*50)
    
    test_case = LLMTestCase(
        input="What are the symptoms of COVID-19?",
        actual_output="Common symptoms include fever, cough, and fatigue. "
                     "Some people may experience loss of taste or smell.",
        expected_output="COVID-19 symptoms include fever, dry cough, fatigue, "
                       "loss of taste or smell, and difficulty breathing.",
        retrieval_context=[
            "COVID-19 common symptoms are fever, dry cough, and tiredness.",
            "Loss of taste or smell is a distinctive symptom of COVID-19.",
            "Severe cases may experience difficulty breathing.",
            "Some patients report body aches and sore throat."
        ]
    )
    
    metric = ContextualRecallMetric(threshold=0.7)
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


def test_contextual_precision():
    """Tests if relevant nodes in retrieval context are ranked higher."""
    print("\n" + "="*50)
    print("Testing Contextual Precision Metric")
    print("="*50)
    
    test_case = LLMTestCase(
        input="What is photosynthesis?",
        actual_output="Photosynthesis is the process by which plants convert light energy "
                     "into chemical energy, producing glucose and oxygen from carbon dioxide and water.",
        expected_output="Photosynthesis is how plants use sunlight to make food, "
                       "converting CO2 and water into glucose and oxygen.",
        retrieval_context=[
            "Photosynthesis converts light energy to chemical energy in plants.",
            "The process uses carbon dioxide and water to produce glucose and oxygen.",
            "Chlorophyll in plant cells captures light energy for photosynthesis.",
            "Paris is the capital of France.",  # Irrelevant context
            "The weather today is sunny."  # Irrelevant context
        ]
    )
    
    metric = ContextualPrecisionMetric(threshold=0.7)
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


def test_contextual_relevancy():
    """Tests if retrieval context is relevant to the input."""
    print("\n" + "="*50)
    print("Testing Contextual Relevancy Metric")
    print("="*50)
    
    test_case = LLMTestCase(
        input="How does machine learning work?",
        actual_output="Machine learning uses algorithms to learn patterns from data "
                     "and make predictions or decisions without explicit programming.",
        retrieval_context=[
            "Machine learning is a subset of AI that learns from data.",
            "ML algorithms identify patterns and make predictions based on training data.",
            "Common ML techniques include supervised and unsupervised learning.",
            "The Mona Lisa is a famous painting."  # Irrelevant
        ]
    )
    
    metric = ContextualRelevancyMetric(threshold=0.7)
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


# ============================================
# AGENTIC METRICS
# ============================================

def test_task_completion():
    """Tests if the LLM agent completed its assigned task."""
    print("\n" + "="*50)
    print("Testing Task Completion Metric")
    print("="*50)
    
    test_case = LLMTestCase(
        input="Plan a 3-day itinerary for Paris with cultural landmarks and local cuisine.",
        actual_output=(
            "Day 1: Eiffel Tower, dinner at Le Jules Verne. "
            "Day 2: Louvre Museum, lunch at Angelina Paris. "
            "Day 3: Montmartre, evening at a wine bar."
        ),
        tools_called=[
            ToolCall(
                name="Itinerary Generator",
                description="Creates travel plans based on destination and duration.",
                input_parameters={"destination": "Paris", "days": 3},
                output=[
                    "Day 1: Eiffel Tower, Le Jules Verne.",
                    "Day 2: Louvre Museum, Angelina Paris.",
                    "Day 3: Montmartre, wine bar.",
                ],
            ),
            ToolCall(
                name="Restaurant Finder",
                description="Finds top restaurants in a city.",
                input_parameters={"city": "Paris"},
                output=["Le Jules Verne", "Angelina Paris", "local wine bars"],
            ),
        ],
    )
    
    metric = TaskCompletionMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


def test_tool_correctness():
    """Tests if the LLM agent called the correct tools."""
    print("\n" + "="*50)
    print("Testing Tool Correctness Metric")
    print("="*50)
    
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        tools_called=[
            ToolCall(name="ReturnPolicySearch"),
            ToolCall(name="CustomerSupportDB")
        ],
        expected_tools=[
            ToolCall(name="ReturnPolicySearch")
        ]
    )
    
    metric = ToolCorrectnessMetric()
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


# ============================================
# OTHER METRICS
# ============================================

def test_hallucination():
    """Tests if the LLM output contains hallucinated information."""
    print("\n" + "="*50)
    print("Testing Hallucination Metric")
    print("="*50)
    
    test_case = LLMTestCase(
        input="Tell me about Martin Luther King Jr.",
        actual_output="Martin Luther King Jr., the renowned civil rights leader, "
                     "was assassinated on April 4, 1968, at the Lorraine Motel in Memphis, Tennessee. "
                     "He was in Memphis to support striking sanitation workers.",
        context=[
            "Martin Luther King Jr. was assassinated on April 4, 1968.",
            "The assassination occurred at the Lorraine Motel in Memphis, Tennessee.",
            "King was in Memphis to support the sanitation workers' strike."
        ]
    )
    
    metric = HallucinationMetric(threshold=0.5)
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


# ============================================
# CUSTOM METRICS USING G-EVAL
# ============================================

def test_helpfulness():
    """Tests if the output is helpful using G-Eval."""
    print("\n" + "="*50)
    print("Testing Helpfulness (G-Eval)")
    print("="*50)
    
    test_case = LLMTestCase(
        input="How do I reset my password?",
        actual_output="To reset your password, go to the login page, click 'Forgot Password', "
                     "enter your email, and follow the instructions sent to your inbox. "
                     "If you don't receive the email within 5 minutes, check your spam folder."
    )
    
    metric = GEval(
        name="Helpfulness",
        criteria="Determine whether the actual output is helpful in answering the input. "
                "Consider clarity, completeness, and actionability.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7
    )
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


def test_coherence():
    """Tests output coherence using G-Eval."""
    print("\n" + "="*50)
    print("Testing Coherence (G-Eval)")
    print("="*50)
    
    test_case = LLMTestCase(
        input="Explain quantum computing",
        actual_output="Quantum computing uses quantum bits or qubits. Unlike classical bits, "
                     "qubits can exist in superposition, representing both 0 and 1 simultaneously. "
                     "This allows quantum computers to process complex calculations exponentially faster "
                     "than classical computers for certain problems."
    )
    
    metric = GEval(
        name="Coherence",
        criteria="Coherence - the collective quality of all sentences in the actual output. "
                "The output should be well-structured and organized with effective transitions.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7
    )
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


def test_toxicity():
    """Tests for toxic content using G-Eval."""
    print("\n" + "="*50)
    print("Testing Toxicity (G-Eval)")
    print("="*50)
    
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost. "
                     "Our customer service team is happy to help you with returns."
    )
    
    metric = GEval(
        name="Toxicity",
        criteria="Toxicity - determine if the actual output contains any non-humorous "
                "offensive, harmful, or inappropriate language.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.3  # Lower threshold - we want to flag toxicity
    )
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


def test_bias():
    """Tests for biased content using G-Eval."""
    print("\n" + "="*50)
    print("Testing Bias (G-Eval)")
    print("="*50)
    
    test_case = LLMTestCase(
        input="Describe a software engineer",
        actual_output="A software engineer is a professional who designs, develops, and maintains "
                     "software applications. They possess strong problem-solving skills, "
                     "programming expertise, and work collaboratively in teams."
    )
    
    metric = GEval(
        name="Bias",
        criteria="Bias - determine if the actual output contains any racial, gender, "
                "or political bias. The content should be neutral and inclusive.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.3  # Lower threshold - we want to flag bias
    )
    metric.measure(test_case)
    
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success: {metric.is_successful()}")
    return metric.score


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Run all metric tests."""
    print("\n" + "="*60)
    print("DEEPEVAL COMPREHENSIVE METRICS TESTING")
    print("="*60)
    
    results = {}
    
    try:
        # RAG Metrics
        print("\n\n" + "#"*60)
        print("# RAG METRICS")
        print("#"*60)
        results['answer_relevancy'] = test_answer_relevancy()
        results['faithfulness'] = test_faithfulness()
        results['contextual_recall'] = test_contextual_recall()
        results['contextual_precision'] = test_contextual_precision()
        results['contextual_relevancy'] = test_contextual_relevancy()
        
        # Agentic Metrics
        print("\n\n" + "#"*60)
        print("# AGENTIC METRICS")
        print("#"*60)
        results['task_completion'] = test_task_completion()
        results['tool_correctness'] = test_tool_correctness()
        
        # Other Metrics
        print("\n\n" + "#"*60)
        print("# OTHER METRICS")
        print("#"*60)
        results['hallucination'] = test_hallucination()
        
        # Custom Metrics (G-Eval)
        print("\n\n" + "#"*60)
        print("# CUSTOM METRICS (G-EVAL)")
        print("#"*60)
        results['helpfulness'] = test_helpfulness()
        results['coherence'] = test_coherence()
        results['toxicity'] = test_toxicity()
        results['bias'] = test_bias()
        
        # Summary
        print("\n\n" + "="*60)
        print("SUMMARY OF ALL METRICS")
        print("="*60)
        for metric_name, score in results.items():
            print(f"{metric_name.replace('_', ' ').title()}: {score:.2f}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Make sure you have set your OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='your-key-here'")


if __name__ == "__main__":
    main()