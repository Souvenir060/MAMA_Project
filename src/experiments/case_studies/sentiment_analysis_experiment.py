#!/usr/bin/env python3
"""
Sentiment Analysis Case Study - Table II
"""

import json
import logging
import numpy as np
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from setfit import SetFitModel, SetFitTrainer
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    import torch
    
    # Set torch seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.error("Please install: pip install setfit datasets torch sentence-transformers")
    sys.exit(1)

class SentimentAnalysisExperiment:
    """
    Real sentiment analysis experiment using SetFit methodology
    """
    
    def __init__(self):
        self.model_name = "all-mpnet-base-v2"  # Same as MAMA framework
        self.results_dir = Path("src/experiments/case_studies/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_sst2_dataset(self) -> Dict[str, Any]:
        """Load Stanford Sentiment Treebank (SST-2) dataset"""
        logger.info("Loading SST-2 dataset...")
        
        try:
            # Load the dataset
            dataset = load_dataset("glue", "sst2")
            
            # Extract validation set (we'll use this for evaluation)
            val_data = dataset['validation']
            
            # Convert to our format
            validation_samples = []
            for item in val_data:
                validation_samples.append({
                    'text': item['sentence'],
                    'label': item['label']  # 0 = negative, 1 = positive
                })
            
            logger.info(f"Loaded {len(validation_samples)} validation samples")
            return {
                'validation': validation_samples,
                'total_val_samples': len(validation_samples)
            }
            
        except Exception as e:
            logger.error(f"Failed to load SST-2 dataset: {e}")
            raise
    
    def create_few_shot_training_set(self, validation_data: List[Dict]) -> List[Dict]:
        """
        Create 16-sample training set (8 positive, 8 negative) from validation data
        As per paper: "We used only 8 positive and 8 negative examples (16 total) for training"
        """
        logger.info("Creating few-shot training set (8 positive + 8 negative samples)...")
        
        # Separate by label
        positive_samples = [s for s in validation_data if s['label'] == 1]
        negative_samples = [s for s in validation_data if s['label'] == 0]
        
        # Randomly select 8 from each class
        selected_positive = random.sample(positive_samples, 8)
        selected_negative = random.sample(negative_samples, 8)
        
        training_set = selected_positive + selected_negative
        random.shuffle(training_set)
        
        logger.info(f"Created training set with {len(training_set)} samples:")
        logger.info(f"  Positive: {sum(1 for s in training_set if s['label'] == 1)}")
        logger.info(f"  Negative: {sum(1 for s in training_set if s['label'] == 0)}")
        
        return training_set
    
    def prepare_evaluation_set(self, validation_data: List[Dict], training_set: List[Dict]) -> List[Dict]:
        """
        Prepare evaluation set by removing training samples from validation set
        """
        # Get texts from training set
        training_texts = {sample['text'] for sample in training_set}
        
        # Remove training samples from validation
        evaluation_set = [
            sample for sample in validation_data 
            if sample['text'] not in training_texts
        ]
        
        logger.info(f"Evaluation set size: {len(evaluation_set)} samples")
        return evaluation_set
    
    def train_setfit_model(self, training_set: List[Dict]) -> SetFitModel:
        """
        Train SetFit model using few-shot training data
        """
        logger.info(f"Training SetFit model with {self.model_name}...")
        
        try:
            # Initialize SetFit model with the same transformer as MAMA
            model = SetFitModel.from_pretrained(self.model_name)
            
            # Prepare training data in SetFit format
            train_texts = [sample['text'] for sample in training_set]
            train_labels = [sample['label'] for sample in training_set]
            
            # Train the model
            logger.info("Starting SetFit training...")
            start_time = time.time()
            
            # SetFit training with few-shot data and required parameters
            model.fit(
                x_train=train_texts, 
                y_train=train_labels,
                num_epochs=1  # Few-shot learning with minimal epochs
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return model
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, model: SetFitModel, evaluation_set: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate trained model on evaluation set
        """
        logger.info(f"Evaluating model on {len(evaluation_set)} samples...")
        
        try:
            # Prepare evaluation data
            eval_texts = [sample['text'] for sample in evaluation_set]
            eval_labels = [sample['label'] for sample in evaluation_set]
            
            # Get predictions
            start_time = time.time()
            predictions = model.predict(eval_texts)
            inference_time = time.time() - start_time
            
            # Calculate accuracy
            correct_predictions = sum(1 for pred, true in zip(predictions, eval_labels) if pred == true)
            total_predictions = len(eval_labels)
            accuracy = correct_predictions / total_predictions
            
            # Prepare detailed results
            detailed_predictions = []
            for i, (text, true_label, pred_label) in enumerate(zip(eval_texts, eval_labels, predictions)):
                detailed_predictions.append({
                    'text': text,
                    'true_label': int(true_label),
                    'predicted_label': int(pred_label),
                    'correct': bool(pred_label == true_label)
                })
            
            results = {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'inference_time_seconds': inference_time,
                'predictions_per_second': total_predictions / inference_time,
                'detailed_predictions': detailed_predictions
            }
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"  Correct: {correct_predictions}/{total_predictions}")
            logger.info(f"  Inference time: {inference_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run the complete sentiment analysis experiment
        """
        logger.info("🚀 Starting Real Sentiment Analysis Experiment (Table II)")
        logger.info("📋 Experiment Configuration:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Method: SetFit Few-Shot Learning")
        logger.info(f"  Training samples: 16 (8 positive + 8 negative)")
        logger.info(f"  Dataset: Stanford Sentiment Treebank (SST-2)")
        
        try:
            # Load dataset
            dataset = self.load_sst2_dataset()
            
            # Create few-shot training set
            training_set = self.create_few_shot_training_set(dataset['validation'])
            
            # Prepare evaluation set
            evaluation_set = self.prepare_evaluation_set(dataset['validation'], training_set)
            
            # Train model
            model = self.train_setfit_model(training_set)
            
            # Evaluate model
            results = self.evaluate_model(model, evaluation_set)
            
            # Package final results
            experiment_results = {
                'experiment_info': {
                    'method': 'SetFit Few-Shot Learning',
                    'model': self.model_name,
                    'training_samples': len(training_set),
                    'evaluation_samples': len(evaluation_set),
                    'dataset': 'Stanford Sentiment Treebank (SST-2)',
                    'timestamp': datetime.now().isoformat(),
                    'random_seed': 42
                },
                'training_data': {
                    'positive_samples': len([s for s in training_set if s['label'] == 1]),
                    'negative_samples': len([s for s in training_set if s['label'] == 0]),
                    'total_samples': len(training_set)
                },
                'results': results
            }
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"sentiment_analysis_real_experiment_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Experiment completed successfully!")
            logger.info(f"📁 Results saved to: {results_file}")
            
            return experiment_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise

def main():
    """Main execution function"""
    try:
        experiment = SentimentAnalysisExperiment()
        results = experiment.run_experiment()
        
        # Print summary for immediate reference
        accuracy = results['results']['accuracy']
        print(f"\n{'='*60}")
        print(f"📊 SENTIMENT ANALYSIS EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"Method: SetFit Few-Shot Learning")
        print(f"Model: all-mpnet-base-v2")
        print(f"Training Samples: 16 (8 positive + 8 negative)")
        print(f"Evaluation Samples: {results['results']['total_predictions']}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"{'='*60}")
        
        return results
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 