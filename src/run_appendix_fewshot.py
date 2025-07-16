#!/usr/bin/env python3
"""
MAMA Framework Sentiment Analysis Case Study - Enhanced Version
Advanced Prompt Engineering with JSON Output and Weighted Voting
Small-scale experiment for paper appendix, demonstrating MAMA framework generalizability to NLP tasks
"""

import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Tuple, Any
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ============================================================================
# Utility Functions
# ============================================================================

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to standard Python types for JSON serialization.
    This fixes the "Object of type float32 is not JSON serializable" error.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def safe_json_parse(response_text: str) -> Dict[str, Any]:
    """
    Safely parse JSON response from LLM, with fallback handling
    """
    try:
        # Try to parse the response as JSON
        result = json.loads(response_text.strip())
        return result
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract JSON from text
        try:
            # Look for JSON-like patterns
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                return result
        except:
            pass
        
        # If all parsing fails, return default structure
        print(f"Warning: Failed to parse JSON response: {response_text}")
        return {"decision": False, "confidence": 0.0}

# ============================================================================
# Configuration
# ============================================================================

# Configure your OpenAI API key here
# Can be set via environment variable or directly
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# LLM Configuration
LLM_MODEL = "gpt-3.5-turbo"  # Can be changed to gpt-4, etc.
MAX_RETRIES = 3
REQUEST_DELAY = 1  # seconds, to avoid API limits

# ============================================================================
# Enhanced Agent Definitions with JSON Output
# ============================================================================

class EnhancedSentimentExpertAgent:
    """Enhanced sentiment analysis expert agent with JSON output"""
    
    def __init__(self, name: str, system_prompt: str, pml_description: str, example_input: str, example_output: str):
        self.name = name
        self.system_prompt = system_prompt
        self.pml_description = pml_description
        self.example_input = example_input
        self.example_output = example_output
    
    def analyze(self, sentence: str) -> Dict[str, Any]:
        """Call LLM for analysis with JSON output"""
        
        # Construct user prompt with few-shot example
        user_prompt = f"""Example Input: '{self.example_input}'
Example Output: {self.example_output}

Now analyze this sentence: '{sentence}'"""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=50
                )
                
                result = response.choices[0].message.content.strip()
                parsed_result = safe_json_parse(result)
                
                # Validate the result structure
                if "decision" in parsed_result and "confidence" in parsed_result:
                    # Ensure confidence is within valid range
                    confidence = float(parsed_result["confidence"])
                    confidence = max(0.0, min(1.0, confidence))
                    
                    return {
                        "decision": bool(parsed_result["decision"]),
                        "confidence": confidence,
                        "raw_response": result
                    }
                else:
                    # Invalid structure, return default
                    return {"decision": False, "confidence": 0.0, "raw_response": result}
                    
            except Exception as e:
                print(f"API call failed for {self.name} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * (attempt + 1))
                else:
                    print(f"Agent {self.name} analysis failed, returning default value")
                    return {"decision": False, "confidence": 0.0, "raw_response": "API_ERROR"}
        
        # Add delay to avoid API limits
        time.sleep(REQUEST_DELAY)
        return {"decision": False, "confidence": 0.0, "raw_response": "TIMEOUT"}

class EnhancedSingleAgentBaseline:
    """Enhanced single agent baseline with JSON output"""
    
    def __init__(self):
        self.system_prompt = "Classify the sentiment of the following sentence. Respond ONLY with a JSON object containing two keys: 'label' (string, 'Positive' or 'Negative') and 'confidence' (a float from 0.0 to 1.0)."
        self.example_input = "This movie is wonderful."
        self.example_output = '{"label": "Positive", "confidence": 0.95}'
    
    def classify(self, sentence: str) -> Tuple[int, Dict[str, Any]]:
        """Direct classification with JSON output, returns 0(negative) or 1(positive)"""
        
        # Construct user prompt with few-shot example
        user_prompt = f"""Example Input: '{self.example_input}'
Example Output: {self.example_output}

Now classify this sentence: '{sentence}'"""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=50
                )
                
                result = response.choices[0].message.content.strip()
                parsed_result = safe_json_parse(result)
                
                # Parse the result
                if "label" in parsed_result:
                    label = parsed_result["label"].lower()
                    confidence = float(parsed_result.get("confidence", 0.5))
                    
                    if "positive" in label:
                        return 1, {"confidence": confidence, "raw_response": result}
                    elif "negative" in label:
                        return 0, {"confidence": confidence, "raw_response": result}
                    else:
                        return 0, {"confidence": 0.0, "raw_response": result}  # Default to negative
                else:
                    return 0, {"confidence": 0.0, "raw_response": result}
                    
            except Exception as e:
                print(f"Baseline API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * (attempt + 1))
                else:
                    return 0, {"confidence": 0.0, "raw_response": "API_ERROR"}
        
        time.sleep(REQUEST_DELAY)
        return 0, {"confidence": 0.0, "raw_response": "TIMEOUT"}

class EnhancedAggregatorAgent:
    """Enhanced rule-based aggregator agent with weighted voting"""
    
    @staticmethod
    def aggregate(selected_agents: List[str], agent_results: Dict[str, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """
        Enhanced fusion rules with weighted voting:
        1. If Sarcasm_Agent is selected and decision is true, final result is Negative(0)
        2. Otherwise, use weighted voting: score = (Positive_decision * confidence) - (Negative_decision * confidence)
        3. Apply negation flip if Negation_Agent detects negation
        """
        
        debug_info = {
            "sarcasm_detected": False,
            "negation_detected": False,
            "weighted_score": 0.0,
            "final_logic": "unknown"
        }
        
        # Rule 1: Sarcasm detection
        if "Sarcasm_Agent" in selected_agents and "Sarcasm_Agent" in agent_results:
            sarcasm_result = agent_results["Sarcasm_Agent"]
            if sarcasm_result.get("decision", False):
                debug_info["sarcasm_detected"] = True
                debug_info["final_logic"] = "sarcasm_override"
                return 0, debug_info  # Sarcasm usually indicates negative sentiment
        
        # Check for negation
        negation_flip = False
        if "Negation_Agent" in selected_agents and "Negation_Agent" in agent_results:
            negation_result = agent_results["Negation_Agent"]
            if negation_result.get("decision", False):
                negation_flip = True
                debug_info["negation_detected"] = True
        
        # Rule 2 & 3: Weighted voting
        weighted_score = 0.0
        
        # Positive agent contribution
        if "Positive_Agent" in agent_results:
            pos_result = agent_results["Positive_Agent"]
            if pos_result.get("decision", False):
                weighted_score += pos_result.get("confidence", 0.0)
            else:
                weighted_score -= pos_result.get("confidence", 0.0) * 0.5  # Partial negative contribution
        
        # Negative agent contribution  
        if "Negative_Agent" in agent_results:
            neg_result = agent_results["Negative_Agent"]
            if neg_result.get("decision", False):
                weighted_score -= neg_result.get("confidence", 0.0)
            else:
                weighted_score += neg_result.get("confidence", 0.0) * 0.5  # Partial positive contribution
        
        debug_info["weighted_score"] = weighted_score
        
        # Apply negation flip if detected
        if negation_flip:
            weighted_score = -weighted_score
            debug_info["final_logic"] = "weighted_voting_with_negation_flip"
        else:
            debug_info["final_logic"] = "weighted_voting"
        
        # Final decision based on weighted score
        final_prediction = 1 if weighted_score > 0 else 0
        
        return final_prediction, debug_info

# ============================================================================
# Enhanced MAMA Framework Core
# ============================================================================

class EnhancedMAMASentimentFramework:
    """Enhanced MAMA framework implementation for sentiment analysis tasks"""
    
    def __init__(self):
        # Initialize sentence embedding model
        print("Loading sentence embedding model...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize enhanced expert agents with few-shot examples
        self.expert_agents = {
            "Positive_Agent": EnhancedSentimentExpertAgent(
                name="Positive_Agent",
                system_prompt="You are an expert in identifying positive language. Analyze the following sentence. Respond ONLY with a JSON object containing two keys: 'decision' (boolean, true if the sentence is positive, otherwise false) and 'confidence' (a float from 0.0 to 1.0 indicating your certainty).",
                pml_description="Specializes in detecting positive emotions, optimistic language, and favorable opinions in text.",
                example_input="This movie is great.",
                example_output='{"decision": true, "confidence": 0.95}'
            ),
            "Negative_Agent": EnhancedSentimentExpertAgent(
                name="Negative_Agent", 
                system_prompt="You are an expert in identifying negative language. Analyze the following sentence. Respond ONLY with a JSON object containing two keys: 'decision' (boolean, true if the sentence is negative, otherwise false) and 'confidence' (a float from 0.0 to 1.0).",
                pml_description="Specializes in detecting negative emotions, pessimistic language, and unfavorable opinions in text.",
                example_input="This movie is terrible.",
                example_output='{"decision": true, "confidence": 0.92}'
            ),
            "Sarcasm_Agent": EnhancedSentimentExpertAgent(
                name="Sarcasm_Agent",
                system_prompt="You are an expert in detecting sarcasm. Analyze the following sentence. Respond ONLY with a JSON object containing two keys: 'decision' (boolean, true if the sentence contains sarcasm, otherwise false) and 'confidence' (a float from 0.0 to 1.0).",
                pml_description="Specializes in detecting sarcasm, irony, and implied meanings that contradict literal interpretation.",
                example_input="Oh great, another boring meeting.",
                example_output='{"decision": true, "confidence": 0.88}'
            ),
            "Negation_Agent": EnhancedSentimentExpertAgent(
                name="Negation_Agent",
                system_prompt="You are a linguistics expert focusing on negation. Analyze the following sentence. Respond ONLY with a JSON object containing two keys: 'decision' (boolean, true if the sentence contains negation words, otherwise false) and 'confidence' (a float from 0.0 to 1.0).",
                pml_description="Specializes in identifying negation words and grammatical constructs that invert or modify sentiment polarity.",
                example_input="This movie is not bad.",
                example_output='{"decision": true, "confidence": 0.98}'
            )
        }
        
        # Pre-compute PML embeddings
        print("Pre-computing PML embeddings...")
        self.pml_embeddings = {}
        for agent_name, agent in self.expert_agents.items():
            self.pml_embeddings[agent_name] = self.embedding_model.encode([agent.pml_description])
        
        # Initialize aggregator
        self.aggregator = EnhancedAggregatorAgent()
        
        print("Enhanced MAMA framework initialized successfully!")
    
    def select_agents(self, sentence: str, top_k: int = 3) -> List[str]:
        """Select Top-K agents based on semantic similarity"""
        
        # Compute input sentence embedding
        sentence_embedding = self.embedding_model.encode([sentence])
        
        # Compute similarity with each agent's PML
        similarities = {}
        for agent_name, pml_embedding in self.pml_embeddings.items():
            similarity = cosine_similarity(sentence_embedding, pml_embedding)[0][0]
            similarities[agent_name] = similarity
        
        # Select Top-K agents
        selected_agents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        selected_agent_names = [agent_name for agent_name, _ in selected_agents]
        
        return selected_agent_names
    
    def classify_sentence(self, sentence: str) -> Tuple[int, Dict[str, Any]]:
        """Classify a single sentence using enhanced MAMA"""
        
        # Step 1: Agent selection
        selected_agents = self.select_agents(sentence, top_k=3)
        
        # Step 2: Agent execution
        agent_results = {}
        for agent_name in selected_agents:
            if agent_name in self.expert_agents:
                result = self.expert_agents[agent_name].analyze(sentence)
                agent_results[agent_name] = result
        
        # Step 3: Enhanced knowledge fusion
        final_prediction, aggregation_debug = self.aggregator.aggregate(selected_agents, agent_results)
        
        # Return prediction and debug info
        debug_info = {
            "selected_agents": selected_agents,
            "agent_results": agent_results,
            "aggregation_debug": aggregation_debug,
            "final_prediction": final_prediction
        }
        
        return final_prediction, debug_info

# ============================================================================
# Main Enhanced Experiment Function
# ============================================================================

def run_enhanced_sentiment_case_study():
    """Run complete enhanced sentiment analysis case study"""
    
    print("="*70)
    print("MAMA Framework Enhanced Sentiment Analysis Case Study")
    print("Advanced Prompt Engineering with JSON Output & Weighted Voting")
    print("="*70)
    
    # Check API key
    if OPENAI_API_KEY == 'your-api-key-here':
        print("Warning: Please configure your OpenAI API key first!")
        print("You can set the OPENAI_API_KEY environment variable or modify the script configuration")
        return
    
    # Step 1: Load dataset
    print("\nStep 1: Loading SST-2 dataset...")
    try:
        dataset = load_dataset('glue', 'sst2')
        validation_data = dataset['validation']
        print(f"Validation set size: {len(validation_data)}")
        print(f"Sample data: {validation_data[0]}")
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return
    
    # Step 2: Initialize enhanced systems
    print("\nStep 2: Initializing enhanced MAMA framework and baseline...")
    mama_framework = EnhancedMAMASentimentFramework()
    baseline_agent = EnhancedSingleAgentBaseline()
    
    # Step 3: Experiment setup (test first 50 samples for demo to save API calls)
    print("\nStep 3: Starting enhanced experiment...")
    test_size = min(50, len(validation_data))  # Limit test size to save API calls
    print(f"Test sample count: {test_size}")
    
    # Store results
    mama_predictions = []
    baseline_predictions = []
    true_labels = []
    detailed_results = []
    
    # Step 4: Execute experiment
    for i in range(test_size):
        sample = validation_data[i]
        sentence = sample['sentence']
        true_label = sample['label']
        
        print(f"\nProcessing sample {i+1}/{test_size}: {sentence[:50]}...")
        
        # MAMA prediction
        try:
            mama_pred, debug_info = mama_framework.classify_sentence(sentence)
            mama_predictions.append(mama_pred)
        except Exception as e:
            print(f"MAMA prediction failed: {e}")
            mama_predictions.append(0)  # Default value
            debug_info = {"error": str(e)}
        
        # Baseline prediction
        try:
            baseline_pred, baseline_debug = baseline_agent.classify(sentence)
            baseline_predictions.append(baseline_pred)
        except Exception as e:
            print(f"Baseline prediction failed: {e}")
            baseline_predictions.append(0)  # Default value
            baseline_debug = {"error": str(e)}
        
        true_labels.append(true_label)
        
        # Record detailed results
        detailed_results.append({
            "sentence": sentence,
            "true_label": true_label,
            "mama_prediction": mama_predictions[-1],
            "baseline_prediction": baseline_predictions[-1],
            "mama_debug_info": debug_info,
            "baseline_debug_info": baseline_debug
        })
        
        # Show progress
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{test_size} samples")
    
    # Step 5: Calculate results
    print("\nStep 5: Computing evaluation metrics...")
    
    # Calculate accuracy
    mama_accuracy = sum(1 for p, t in zip(mama_predictions, true_labels) if p == t) / len(true_labels)
    baseline_accuracy = sum(1 for p, t in zip(baseline_predictions, true_labels) if p == t) / len(true_labels)
    
    # Step 6: Output results
    print("\n" + "="*70)
    print("Enhanced Experiment Results")
    print("="*70)
    
    print(f"\nTest sample count: {test_size}")
    print(f"MAMA framework accuracy: {mama_accuracy:.4f} ({mama_accuracy*100:.2f}%)")
    print(f"Single agent baseline accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"Accuracy improvement: {(mama_accuracy - baseline_accuracy)*100:.2f} percentage points")
    
    # Generate Markdown table
    print("\n" + "="*70)
    print("Table for Paper Appendix (Markdown Format)")
    print("="*70)
    
    markdown_table = f"""
| Method | Accuracy | Sample Count | Dataset |
|--------|----------|--------------|---------|
| MAMA Framework (Enhanced) | {mama_accuracy:.4f} ({mama_accuracy*100:.2f}%) | {test_size} | SST-2 validation |
| Single Agent Baseline (Enhanced) | {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%) | {test_size} | SST-2 validation |
| **Improvement** | **{(mama_accuracy - baseline_accuracy)*100:+.2f} percentage points** | - | - |

**Experimental Configuration**:
- **Dataset**: Stanford Sentiment Treebank (SST-2) validation set  
- **Task**: Binary sentiment classification (0: Negative, 1: Positive)  
- **Expert Agents**: Positive_Agent, Negative_Agent, Sarcasm_Agent, Negation_Agent  
- **Selection Strategy**: Top-3 based on semantic similarity (all-mpnet-base-v2)  
- **Fusion Strategy**: Enhanced weighted voting with confidence scores  
- **Output Format**: JSON with decision (boolean) and confidence (float)  
- **Prompt Engineering**: Few-shot examples for robustness  
- **LLM Model**: GPT-3.5-turbo  
- **API Calls**: Fully real, no simulation  
"""
    
    print(markdown_table)
    
    # Save detailed results
    print("\nStep 6: Saving enhanced detailed results...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"enhanced_sentiment_case_study_results_{timestamp}.json"
    
    results_data = {
        "experiment_info": {
            "version": "enhanced_v2",
            "dataset": "SST-2",
            "split": "validation",
            "test_size": test_size,
            "timestamp": timestamp,
            "improvements": [
                "JSON output format with confidence scores",
                "Few-shot prompt engineering",
                "Weighted voting fusion strategy",
                "Enhanced error handling and parsing",
                "Updated OpenAI API v1.0+ compatibility"
            ]
        },
        "metrics": {
            "mama_accuracy": mama_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "improvement": mama_accuracy - baseline_accuracy
        },
        "detailed_results": detailed_results
    }
    
    # Convert numpy types before saving to JSON
    results_data_serializable = convert_numpy_types(results_data)
    
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"Enhanced detailed results saved to: {results_filename}")
    
    print("\n" + "="*70)
    print("Enhanced case study completed successfully!")
    print("="*70)

if __name__ == "__main__":
    run_enhanced_sentiment_case_study() 