#!/usr/bin/env python3
"""
MAMA Framework Sentiment Analysis Case Study
"""

import os
import json
import time
from typing import Dict, List, Tuple, Any
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
from datetime import datetime
from pathlib import Path

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
    Safely parse JSON response from simulated LLM, with fallback handling
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

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Maximum number of samples for the experiment (to control execution time)
MAX_SAMPLES = 50

# Simulation delay to mimic real API calls (set to 0 for fastest execution)
SIMULATION_DELAY = 0.05  # seconds

# ============================================================================
# Offline Agent Definitions with Simulated Responses
# ============================================================================

class OfflineSentimentExpertAgent:
    """Offline sentiment analysis expert agent with simulated responses"""
    
    def __init__(self, name: str, system_prompt: str, pml_description: str, example_input: str, example_output: str):
        self.name = name
        self.system_prompt = system_prompt
        self.pml_description = pml_description
        self.example_input = example_input
        self.example_output = example_output
        
        # Rules for response generation
        self.keyword_patterns = {
            "positive_agent": {
                "positive_words": ["good", "great", "excellent", "amazing", "wonderful", "love", "beautiful", "enjoy", "happy", "like"],
                "threshold": 0.6,
                "confidence_range": (0.65, 0.95)
            },
            "negative_agent": {
                "negative_words": ["bad", "terrible", "awful", "horrible", "hate", "dislike", "poor", "disappointing", "worst", "boring"],
                "threshold": 0.6,
                "confidence_range": (0.65, 0.95)
            },
            "sarcasm_agent": {
                "sarcasm_patterns": ["but ", " but", "however", "though", "supposed to", "sure", "right", "clearly", "obviously"],
                "mixed_sentiment": ["good", "bad", "great", "terrible"],
                "threshold": 0.4,
                "confidence_range": (0.60, 0.90)
            },
            "negation_agent": {
                "negation_words": ["not", "no ", "never", "don't", "doesn't", "isn't", "wasn't", "weren't", "haven't", "hasn't", "won't", "wouldn't", "shouldn't", "couldn't", "can't"],
                "threshold": 0.7,
                "confidence_range": (0.70, 0.95)
            }
        }
    
    def analyze(self, sentence: str) -> Dict[str, Any]:
        """Simulate LLM for analysis with deterministic simulated response"""
        
        # Add delay to simulate API call
        time.sleep(SIMULATION_DELAY)
        
        # Get the agent type from name
        agent_type = self.name.lower().replace(" ", "_")
        
        # Deterministic response generation based on input
        sentence_lower = sentence.lower()
        
        # Get the rules for this agent
        rules = self.keyword_patterns.get(agent_type, {})
        if not rules:
            # Default behavior if agent type not recognized
            return {"decision": False, "confidence": 0.5, "raw_response": '{"decision": false, "confidence": 0.5}'}
        
        # Generate response based on agent-specific rules
        if agent_type == "positive_agent":
            # Check for positive words
            word_matches = sum(1 for word in rules["positive_words"] if word in sentence_lower)
            match_ratio = word_matches / len(rules["positive_words"])
            
            decision = match_ratio >= rules["threshold"]
            # Make confidence proportional to match ratio but within range
            confidence_min, confidence_max = rules["confidence_range"]
            confidence = confidence_min + (confidence_max - confidence_min) * match_ratio
            
            if "not " in sentence_lower or "n't " in sentence_lower:
                # Reduce confidence if negation is present
                confidence *= 0.7
                
        elif agent_type == "negative_agent":
            # Check for negative words
            word_matches = sum(1 for word in rules["negative_words"] if word in sentence_lower)
            match_ratio = word_matches / len(rules["negative_words"])
            
            decision = match_ratio >= rules["threshold"]
            # Make confidence proportional to match ratio but within range
            confidence_min, confidence_max = rules["confidence_range"]
            confidence = confidence_min + (confidence_max - confidence_min) * match_ratio
            
        elif agent_type == "sarcasm_agent":
            # Check for sarcasm patterns
            pattern_matches = sum(1 for pattern in rules["sarcasm_patterns"] if pattern in sentence_lower)
            has_mixed = any(word in sentence_lower for word in rules["mixed_sentiment"])
                
            # Sarcasm detection requires both patterns and mixed sentiment
            decision = (pattern_matches > 0) and has_mixed
            confidence_min, confidence_max = rules["confidence_range"]
            match_score = min(1.0, pattern_matches / len(rules["sarcasm_patterns"]))
            confidence = confidence_min + (confidence_max - confidence_min) * match_score
            
        elif agent_type == "negation_agent":
            # Check for negation words
            negation_matches = sum(1 for word in rules["negation_words"] if word in sentence_lower)
            
            # Decision if any negation is present
            decision = negation_matches > 0
            confidence_min, confidence_max = rules["confidence_range"]
            match_score = min(1.0, negation_matches / len(rules["negation_words"]))
            confidence = confidence_min + (confidence_max - confidence_min) * match_score
            
        # Format as JSON string
        raw_response = json.dumps({"decision": decision, "confidence": round(confidence, 2)})
                    
        return {
            "decision": decision,
                        "confidence": confidence,
            "raw_response": raw_response
        }
                    
class OfflineSingleAgentBaseline:
    """Offline single agent baseline with simulated responses"""
    
    def __init__(self):
        self.system_prompt = "Classify the sentiment of the following sentence."
        self.example_input = "This movie is wonderful."
        self.example_output = '{"label": "Positive", "confidence": 0.95}'
        
        # Simple positive/negative word lists
        self.positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "love", "beautiful", 
            "enjoy", "happy", "like", "best", "fantastic", "perfect", "brilliant",
            "outstanding", "superb", "awesome", "delightful", "pleasant", "remarkable"
        ]
        self.negative_words = [
            "bad", "terrible", "awful", "horrible", "hate", "dislike", "poor", 
            "disappointing", "worst", "boring", "stupid", "waste", "pathetic",
            "disaster", "miserable", "annoying", "dreadful", "mediocre", "lousy", "failure"
        ]
    
    def classify(self, sentence: str) -> Tuple[int, Dict[str, Any]]:
        """Direct classification with JSON output, returns 0(negative) or 1(positive)"""
        
        # Add delay to simulate API call
        time.sleep(SIMULATION_DELAY)
        
        # Simple rule-based sentiment analysis
        sentence_lower = sentence.lower()
        
        # Count word occurrences
        pos_count = sum(1 for word in self.positive_words if word in sentence_lower)
        neg_count = sum(1 for word in self.negative_words if word in sentence_lower)
        
        # Check for negations which could flip sentiment
        negation_words = ["not", "no", "never", "don't", "doesn't", "isn't", "wasn't"]
        has_negation = any(neg in sentence_lower for neg in negation_words)
        
        if has_negation:
            # Simple negation handling - flip the counts
            pos_count, neg_count = neg_count, pos_count
            
        # Determine sentiment and confidence
        if pos_count > neg_count:
            label = "Positive"
            confidence = min(0.95, max(0.6, 0.6 + 0.35 * (pos_count / (pos_count + neg_count + 0.1))))
            result = 1
        else:
            label = "Negative"
            confidence = min(0.95, max(0.6, 0.6 + 0.35 * (neg_count / (pos_count + neg_count + 0.1))))
            result = 0
            
        # Format as JSON string
        response_text = json.dumps({"label": label, "confidence": round(confidence, 2)})
        
        return result, {"confidence": confidence, "raw_response": response_text}

class OfflineAggregatorAgent:
    """Aggregates outputs from specialized sentiment agents using rules"""
    
    @staticmethod
    def aggregate(selected_agents: List[str], agent_results: Dict[str, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """
        Rule-based aggregation of agent outputs
        
        Args:
            selected_agents: List of agent IDs that were selected
            agent_results: Dictionary of results from each agent
            
        Returns:
            Tuple of (sentiment label as int, metadata dict)
        """
        # Rule 1: Sarcasm detection takes precedence
        if 'sarcasm_agent' in selected_agents and agent_results['sarcasm_agent']['decision']:
            sarcasm_confidence = agent_results['sarcasm_agent']['confidence']
            if sarcasm_confidence > 0.7:
                # High confidence sarcasm usually indicates negative sentiment
                return 0, {
                    'rule_applied': 'sarcasm_detection',
                    'confidence': sarcasm_confidence,
                    'explanation': 'Sarcasm detected, classified as negative'
                }
        
        # Rule 2: Negation handling
        if 'negation_agent' in selected_agents and agent_results['negation_agent']['decision']:
            # Get negation confidence
            negation_confidence = agent_results['negation_agent']['confidence']
            
            # If negation detected with high confidence, check positive/negative signals
            if negation_confidence > 0.7:
                # Get positive/negative signals
                has_positive = 'positive_agent' in selected_agents and agent_results['positive_agent']['decision']
                has_negative = 'negative_agent' in selected_agents and agent_results['negative_agent']['decision']
                
                if has_positive and not has_negative:
                    # Negation of positive is negative
                    return 0, {
                        'rule_applied': 'negation_of_positive',
                        'confidence': negation_confidence * agent_results['positive_agent']['confidence'],
                        'explanation': 'Negation of positive sentiment detected'
                    }
                elif has_negative and not has_positive:
                    # Negation of negative is positive
                    return 1, {
                        'rule_applied': 'negation_of_negative',
                        'confidence': negation_confidence * agent_results['negative_agent']['confidence'],
                        'explanation': 'Negation of negative sentiment detected'
                    }
        
        # Rule 3: Weighted voting for remaining cases
        pos_signal = 0
        neg_signal = 0
        
        if 'positive_agent' in selected_agents:
            pos_weight = agent_results['positive_agent']['confidence'] if agent_results['positive_agent']['decision'] else 0
            pos_signal = pos_weight
        
        if 'negative_agent' in selected_agents:
            neg_weight = agent_results['negative_agent']['confidence'] if agent_results['negative_agent']['decision'] else 0
            neg_signal = neg_weight
            
        # Make the decision
        if pos_signal > neg_signal:
            confidence = pos_signal / (pos_signal + neg_signal + 0.001)
            return 1, {
                'rule_applied': 'weighted_voting',
                'confidence': float(confidence),
                'positive_signal': float(pos_signal),
                'negative_signal': float(neg_signal),
                'explanation': 'Positive sentiment prevails in weighted voting'
            }
        else:
            confidence = neg_signal / (pos_signal + neg_signal + 0.001)
            return 0, {
                'rule_applied': 'weighted_voting',
                'confidence': float(confidence),
                'positive_signal': float(pos_signal),
                'negative_signal': float(neg_signal),
                'explanation': 'Negative sentiment prevails in weighted voting'
            }

class OfflineMAMASentimentFramework:
    """MAMA Framework adapted for sentiment analysis task - Offline Version"""
    
    def __init__(self):
        """Initialize the MAMA sentiment framework"""
        
        # Initialize sentence embedding model
        try:
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Loaded SBERT model: all-MiniLM-L6-v2")
        except Exception as e:
            print(f"⚠ Failed to load SBERT model: {e}")
            print("⚠ Using fallback random embeddings")
            self.sbert_model = None
        
        # Define agents with PML descriptions
        self.agents = {
            'positive_agent': OfflineSentimentExpertAgent(
                name="Positive Agent", 
                system_prompt="You are an expert at identifying positive sentiment in text. Analyze the input and determine if it contains positive sentiment.",
                pml_description="<pml:expert domain='sentiment_analysis' specialty='positive_sentiment_detection'>\nIdentifies positive emotions, praise, approval, and satisfaction in text.\nSpecializes in detecting subtle positive cues, enthusiastic language, and words of appreciation.\nTrained on consumer reviews, social media posts, and professional evaluations.\n</pml:expert>",
                example_input="This movie is absolutely wonderful and I enjoyed every minute.",
                example_output='{"decision": true, "confidence": 0.95}'
            ),
            'negative_agent': OfflineSentimentExpertAgent(
                name="Negative Agent", 
                system_prompt="You are an expert at identifying negative sentiment in text. Analyze the input and determine if it contains negative sentiment.",
                pml_description="<pml:expert domain='sentiment_analysis' specialty='negative_sentiment_detection'>\nIdentifies negative emotions, criticism, disappointment, and disapproval in text.\nSpecializes in detecting subtle negative cues, complaints, and expressions of dissatisfaction.\nTrained on consumer complaints, critical reviews, and problem reports.\n</pml:expert>",
                example_input="The service was terrible and I will never go back.",
                example_output='{"decision": true, "confidence": 0.92}'
            ),
            'sarcasm_agent': OfflineSentimentExpertAgent(
                name="Sarcasm Agent", 
                system_prompt="You are an expert at identifying sarcasm in text. Analyze the input and determine if it contains sarcasm.",
                pml_description="<pml:expert domain='sentiment_analysis' specialty='sarcasm_detection'>\nIdentifies sarcasm, irony, and statements where literal meaning differs from intended meaning.\nSpecializes in detecting context mismatches, hyperbole used for ironic effect, and mock praise.\nTrained on social media posts, comedy writing, and conversational datasets with labeled sarcasm.\n</pml:expert>",
                example_input="Oh sure, that's EXACTLY what I wanted to happen.",
                example_output='{"decision": true, "confidence": 0.87}'
            ),
            'negation_agent': OfflineSentimentExpertAgent(
                name="Negation Agent", 
                system_prompt="You are an expert at identifying negation in text. Analyze the input and determine if it contains language negation.",
                pml_description="<pml:expert domain='sentiment_analysis' specialty='negation_detection'>\nIdentifies linguistic negations that can reverse sentiment polarity.\nSpecializes in detecting negative particles (not, never), negative prefixes, and implicit negations.\nTrained on syntactic parsing tasks and sentiment-annotated text with negation markers.\n</pml:expert>",
                example_input="I don't think this restaurant is good at all.",
                example_output='{"decision": true, "confidence": 0.96}'
            ),
        }
        
        # Initialize baseline single agent
        self.baseline_agent = OfflineSingleAgentBaseline()
        
        # Initialize aggregator
        self.aggregator = OfflineAggregatorAgent()
    
    def select_agents(self, sentence: str, top_k: int = 3) -> List[str]:
        """Select the most relevant agents using semantic similarity to the sentence"""
        
        # Calculate embeddings
        if self.sbert_model:
            # Using real SBERT
            sentence_embedding = self.sbert_model.encode(sentence)
        
            # Calculate similarity with each agent's PML description
        similarities = {}
        for agent_id, agent in self.agents.items():
                agent_embedding = self.sbert_model.encode(agent.pml_description)
                similarity = cosine_similarity([sentence_embedding], [agent_embedding])[0][0]
                similarities[agent_id] = similarity
        
            # Select top k agents
                selected_agents = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)[:top_k]
                return selected_agents
        else:
            # Fallback: return a deterministic selection based on sentence length
            sentence_hash = sum(ord(c) for c in sentence) % 3
            if sentence_hash == 0:
                return ['positive_agent', 'negative_agent', 'negation_agent']
            elif sentence_hash == 1:
                return ['negative_agent', 'sarcasm_agent', 'negation_agent']
            else:
                return ['positive_agent', 'sarcasm_agent', 'negative_agent']
    
    def classify_sentence(self, sentence: str) -> Tuple[int, Dict[str, Any]]:
        """Classify a sentence using MAMA framework"""
        
        # Step 1: Select agents based on sentence
        selected_agents = self.select_agents(sentence)
        
        # Step 2: Get analysis from each selected agent
        agent_results = {}
        for agent_id in selected_agents:
            agent_results[agent_id] = self.agents[agent_id].analyze(sentence)
        
        # Step 3: Aggregate results
        sentiment, metadata = self.aggregator.aggregate(selected_agents, agent_results)
        
        # Add selected agents to metadata
        metadata['selected_agents'] = selected_agents
        metadata['agent_results'] = agent_results
        
        return sentiment, metadata

def run_offline_sentiment_case_study():
    """Run the full offline sentiment analysis case study"""
    
    print("=" * 80)
    print("MAMA Framework - Sentiment Analysis Case Study (Offline Version)")
    print("=" * 80)
    print("Loading data and models...")
    
    # Load SST-2 dataset
    try:
        # Load the full dataset
        sst2_dataset = load_dataset("stanfordnlp/sst2")
        
        # Get validation split
        validation_data = sst2_dataset["validation"]
        
        # Limit number of samples for faster execution
        sample_indices = list(range(len(validation_data)))
        random.seed(RANDOM_SEED)
        random.shuffle(sample_indices)
        sample_indices = sample_indices[:MAX_SAMPLES]
        
        # Create subsample
        sentences = [validation_data[i]["sentence"] for i in sample_indices]
        labels = [validation_data[i]["label"] for i in sample_indices]
        
        print(f"✓ Loaded {len(sentences)} samples from SST-2 validation set")
        
    except Exception as e:
        print(f"⚠ Error loading SST-2 dataset: {e}")
        print("⚠ Using fallback dataset")
        
        # Fallback dataset - small hand-created
        sentences = [
            "This movie was really good.",
            "I hated everything about it.",
            "The actors were great but the plot was terrible.",
            "It wasn't as bad as people said.",
            "This is supposed to be their best work?",
            "Not a bad way to spend an evening.",
            "I don't think I've ever seen anything worse.",
            "It's not great but it's not terrible either.",
            "I really enjoyed the experience.",
            "Absolutely the worst movie of the year."
        ]
        # 1 is positive, 0 is negative
        labels = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
        print(f"✓ Using fallback dataset with {len(sentences)} samples")
    
    # Initialize the MAMA framework and baseline
    mama_framework = OfflineMAMASentimentFramework()
    baseline_agent = OfflineSingleAgentBaseline()
    
    print("Starting evaluation...")
    
    # Track results
    results = {
        "baseline": {
            "correct": 0,
            "incorrect": 0,
            "predictions": []
        },
        "mama": {
            "correct": 0,
            "incorrect": 0,
            "predictions": []
        },
        "raw_data": []
    }
    
    # Process each sentence
    for i, (sentence, label) in enumerate(zip(sentences, labels)):
        print(f"\nProcessing {i+1}/{len(sentences)}: '{sentence[:50]}...' if longer")
        
        # Baseline classification
        baseline_pred, baseline_meta = baseline_agent.classify(sentence)
        baseline_correct = (baseline_pred == label)
        
        # MAMA framework classification
        mama_pred, mama_meta = mama_framework.classify_sentence(sentence)
        mama_correct = (mama_pred == label)
        
        # Print results
        print(f"  ⊛ Ground truth: {label} ({'Positive' if label == 1 else 'Negative'})")
        print(f"  ⊙ Baseline: {baseline_pred} ({'✓' if baseline_correct else '✗'}) - Conf: {baseline_meta.get('confidence', 'N/A'):.2f}")
        print(f"  ⊛ MAMA: {mama_pred} ({'✓' if mama_correct else '✗'}) - Rule: {mama_meta.get('rule_applied', 'N/A')}")
        print(f"      Selected agents: {mama_meta['selected_agents']}")
        
        # Update results
        if baseline_correct:
            results["baseline"]["correct"] += 1
        else:
            results["baseline"]["incorrect"] += 1
            
        if mama_correct:
            results["mama"]["correct"] += 1
        else:
            results["mama"]["incorrect"] += 1
            
        # Save prediction details
        results["baseline"]["predictions"].append({
            "prediction": baseline_pred,
            "confidence": baseline_meta.get("confidence", 0.0),
            "correct": baseline_correct
        })
        
        results["mama"]["predictions"].append({
            "prediction": mama_pred,
            "metadata": mama_meta,
            "correct": mama_correct
        })
        
        # Save full case data
        results["raw_data"].append({
            "id": i,
            "sentence": sentence,
            "ground_truth": label,
            "baseline": {
                "prediction": baseline_pred,
                "metadata": baseline_meta,
                "correct": baseline_correct
            },
            "mama": {
                "prediction": mama_pred,
                "metadata": mama_meta,
                "correct": mama_correct
            }
        })
    
    # Calculate final metrics
    baseline_accuracy = results["baseline"]["correct"] / len(sentences)
    mama_accuracy = results["mama"]["correct"] / len(sentences)
    improvement = mama_accuracy - baseline_accuracy
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(sentences)}")
    print(f"Baseline accuracy: {baseline_accuracy:.4f} ({results['baseline']['correct']}/{len(sentences)})")
    print(f"MAMA accuracy: {mama_accuracy:.4f} ({results['mama']['correct']}/{len(sentences)})")
    print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    
    # Generate markdown table for paper
    markdown_table = f"""
| Method | Accuracy | Samples |
|--------|----------|---------|
| MAMA Framework | {mama_accuracy:.4f} ({mama_accuracy*100:.2f}%) | {len(sentences)} |
| Single Agent Baseline | {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%) | {len(sentences)} |
| **Improvement** | **{improvement:.4f} ({improvement*100:.2f}%)** | - |
"""
    print("\nPaper Results Table (Markdown):")
    print(markdown_table)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"sentiment_case_study_results_{timestamp}.json"
    
    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save results to file
    with open(results_dir / results_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(results), f, indent=2, ensure_ascii=False)
    
    print(f"\nComplete results saved to results/{results_file}")
    print("\nExperiment completed successfully!")
    
    return results

if __name__ == "__main__":
    run_offline_sentiment_case_study() 