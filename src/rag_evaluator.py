from typing import List, Optional
import os
#from anthropic import Anthropic
from ollama import chat
from ollama import ChatResponse
from ollama import Client
from pydantic import BaseModel
import json

class Score(BaseModel):
  score: Optional[int]

def evaluate_rag_relevance(topic: str, retrievals: List[str]) -> Optional[List[int]]:
    """
    Evaluates the relevance between a topic and a list of retrieved passages using LLM.
    
    Args:
        topic (str): The topic or query string
        retrievals (List[str]): List of retrieved passages to evaluate
        
    Returns:
        Optional[List[int]]: List of relevance scores (0-10) for each retrieval,
                            or None if retrievals list is empty
    """
    # Handle empty retrievals case
    if not retrievals:
        return None
        
    # Initialize Anthropic client
    #client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    client = Client(
        host='http://localhost:11434',
    )
    
    scores = []
    
    for retrieval in retrievals:
        # Construct prompt for the LLM
        prompt = f"""
        On a scale of 0-100, evaluate how relevant the following passage is to the given topic.
        Score guidelines:
        - 0-20: Not relevant at all
        - 21-40: Slightly relevant
        - 41-60: Moderately relevant
        - 61-80: Very relevant
        - 81-100: Highly relevant and directly addresses the topic
        
        Topic: {topic}
        
        Passage: {retrieval}
        
        Please respond with only a single integer score between 0 and 100.
        """
        
        try:
            # Make API call to Claude
            #response = client.messages.create(
            #    model="claude-3-sonnet-20240229",
            response: ChatResponse = client.chat(
                model='llama3.2:3b-instruct-q8_0',
                #model='aya-expanse:8b-q8_0', 
                #max_tokens=10,
                #temperature=0,
                options={'temperature': 0, 
                         'max_tokens': 10
                         },
                format=Score.model_json_schema(),
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract and validate score
            #print(type(json.loads(response['message']['content'])))
            #score = int(response['message']['content'].strip())
            score = json.loads(response['message']['content'])["score"]
            score = max(0, min(100, score))  # Ensure score is between 0-100
            scores.append(score)
            
        except Exception as e:
            print(f"Error evaluating retrieval: {e}")
            scores.append(0)  # Default to 0 score on error
            
    return scores

