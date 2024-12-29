import os
from typing import List
from rag_evaluator import evaluate_rag_relevance  # Assuming previous code is saved as evaluate_rag.py

def run_test_cases():
    """
    Run various test cases to verify the RAG evaluation functionality
    """
    # Test Case 1: Normal case with varied relevance
    print("\nTest Case 1: Normal case with varied relevance")
    topic1 = "Benefits of regular exercise"
    retrievals1 = [
        "Regular physical activity can improve your muscle strength and boost your endurance. Exercise delivers oxygen and nutrients to your tissues and helps your cardiovascular system work more efficiently.",
        "The price of gold has been fluctuating in recent months due to global economic factors.",
        "Studies show that exercise can reduce anxiety, depression, and improve mood by releasing endorphins."
    ]
    scores1 = evaluate_rag_relevance(topic1, retrievals1)
    print(f"Topic: {topic1}")
    for i, (retrieval, score) in enumerate(zip(retrievals1, scores1)):
        print(f"\nRetrieval {i+1}:\n{retrieval[:100]}...")
        print(f"Score: {score}")

    # Test Case 2: Empty retrievals list
    print("\nTest Case 2: Empty retrievals list")
    topic2 = "Artificial Intelligence"
    retrievals2 = []
    scores2 = evaluate_rag_relevance(topic2, retrievals2)
    print(f"Topic: {topic2}")
    print(f"Retrievals: {retrievals2}")
    print(f"Expected: None")
    print(f"Result: {scores2}")

    # Test Case 3: Single retrieval
    print("\nTest Case 3: Single highly relevant retrieval")
    topic3 = "Climate change"
    retrievals3 = [
        "Global warming is causing significant changes to Earth's climate systems, leading to rising sea levels, extreme weather events, and disruption of ecosystems worldwide."
    ]
    scores3 = evaluate_rag_relevance(topic3, retrievals3)
    print(f"Topic: {topic3}")
    print(f"Retrieval: {retrievals3[0][:100]}...")
    print(f"Score: {scores3[0]}")

    # Test Case 4: Very short retrievals
    print("\nTest Case 4: Very short retrievals")
    topic4 = "Space exploration"
    retrievals4 = [
        "NASA launched a new mission.",
        "Weather forecast for tomorrow.",
        "Mars rover discovers evidence."
    ]
    scores4 = evaluate_rag_relevance(topic4, retrievals4)
    print(f"Topic: {topic4}")
    for i, (retrieval, score) in enumerate(zip(retrievals4, scores4)):
        print(f"\nRetrieval {i+1}: {retrieval}")
        print(f"Score: {score}")

def verify_scores(scores: List[int]) -> bool:
    """
    Verify that scores meet basic requirements
    """
    if not scores:
        return False
    return all(isinstance(score, int) and 0 <= score <= 100 for score in scores)

def main():
    # Check for API key
    #if not os.environ.get("ANTHROPIC_API_KEY"):
    #    print("Error: ANTHROPIC_API_KEY environment variable not set")
    #    return

    try:
        run_test_cases()
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
