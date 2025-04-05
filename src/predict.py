import os
import torch
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import model components
from flight_delay_model import (
    FlightDataPreprocessor,
    SimAMCNNMogrifierLSTM
)

class FlightDelayPredictor:
    """
    Class for predicting flight delays using the trained SimAM-CNN-MogrifierLSTM model.
    """
    def __init__(self, model_path, preprocessor=None, device=None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model
            preprocessor: Trained FlightDataPreprocessor (if None, will attempt to load from model_path directory)
            device: Device to run predictions on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.preprocessor = preprocessor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load the preprocessor
        if self.preprocessor is None:
            preprocessor_path = os.path.join(os.path.dirname(model_path), 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                import pickle
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
            else:
                raise ValueError("Preprocessor not provided and not found in model directory")
        
        # Load model configuration
        config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                'input_size': self.preprocessor.total_dim,
                'hidden_size': 128,
                'num_classes': 5,
                'cnn_channels': [32, 64],
                'kernel_sizes': [1, 3],
                'use_simam': True,
                'lambda_val': 0.1,
                'num_rounds': 6
            }
        
        # Initialize model
        self.model = self._initialize_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Define delay categories
        self.delay_categories = [
            'On Time',
            'Slight Delay (1-15 min)',
            'Minor Delay (16-30 min)',
            'Moderate Delay (31-60 min)',
            'Severe Delay (>60 min)'
        ]
    
    def _initialize_model(self):
        """Initialize and load the trained model."""
        model = SimAMCNNMogrifierLSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_classes=self.config['num_classes'],
            cnn_channels=self.config['cnn_channels'],
            kernel_sizes=self.config['kernel_sizes'],
            use_simam=self.config['use_simam'],
            lambda_val=self.config['lambda_val'],
            num_rounds=self.config['num_rounds']
        )
        
        # Load model weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        return model
    
    def predict_delay(self, flight_chain):
        """
        Predict delay level for a flight chain.
        
        Args:
            flight_chain: List of dicts containing flight data for a sequence
                Each dict should have keys matching the preprocessor's expected features
            
        Returns:
            Dictionary with predicted delay level, category, and probabilities
        """
        self.model.eval()
        
        # Preprocess flight chain
        X = self.preprocessor.transform_flight_chain(flight_chain)
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        delay_level = predicted.item()
        probs = probabilities.cpu().numpy()[0]
        
        return {
            'delay_level': delay_level,
            'delay_category': self.delay_categories[delay_level],
            'probabilities': {cat: float(prob) for cat, prob in zip(self.delay_categories, probs)}
        }
    
    def predict_batch(self, flight_chains):
        """
        Predict delays for a batch of flight chains.
        
        Args:
            flight_chains: List of flight chains, where each chain is a list of flight dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        
        # Preprocess all flight chains
        X_batch = []
        for chain in flight_chains:
            X = self.preprocessor.transform_flight_chain(chain)
            X_batch.append(X)
        
        X_tensor = torch.FloatTensor(X_batch).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Format results
        predictions = []
        for i in range(len(flight_chains)):
            delay_level = predicted[i].item()
            probs = probabilities[i].cpu().numpy()
            
            predictions.append({
                'delay_level': delay_level,
                'delay_category': self.delay_categories[delay_level],
                'probabilities': {cat: float(prob) for cat, prob in zip(self.delay_categories, probs)}
            })
        
        return predictions
    
    def analyze_feature_importance(self, flight_chain, num_perturbations=10, perturbation_scale=0.1):
        """
        Simple analysis of feature importance by perturbing inputs.
        
        Args:
            flight_chain: Flight chain to analyze
            num_perturbations: Number of perturbations per feature
            perturbation_scale: Scale of perturbations
            
        Returns:
            DataFrame with feature importance scores
        """
        self.model.eval()
        
        # Preprocess flight chain
        X = self.preprocessor.transform_flight_chain(flight_chain)
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
        
        # Get baseline prediction
        with torch.no_grad():
            outputs = self.model(X_tensor)
            baseline_probs = torch.nn.functional.softmax(outputs, dim=1)
            baseline_pred = torch.argmax(baseline_probs, dim=1).item()
        
        # Initialize feature importance scores
        feature_importance = {}
        
        # Perturb each feature
        for flight_idx in range(X.shape[0]):
            for feature_idx in range(X.shape[1]):
                feature_name = f"Flight {flight_idx+1}, Feature {feature_idx+1}"
                
                # Skip if feature value is zero (might be one-hot encoded)
                if X[flight_idx, feature_idx] == 0:
                    continue
                
                changes = []
                
                # Perturb the feature multiple times
                for _ in range(num_perturbations):
                    # Create a perturbed copy
                    X_perturbed = X.copy()
                    
                    # Apply perturbation
                    perturbation = np.random.normal(0, perturbation_scale * abs(X[flight_idx, feature_idx]))
                    X_perturbed[flight_idx, feature_idx] += perturbation
                    
                    # Convert to tensor
                    X_perturbed_tensor = torch.FloatTensor(X_perturbed).unsqueeze(0).to(self.device)
                    
                    # Get prediction for perturbed input
                    with torch.no_grad():
                        outputs = self.model(X_perturbed_tensor)
                        perturbed_probs = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Calculate change in probability for the baseline class
                    prob_change = abs(float(perturbed_probs[0, baseline_pred] - baseline_probs[0, baseline_pred]))
                    changes.append(prob_change)
                
                # Average change across perturbations
                feature_importance[feature_name] = np.mean(changes)
        
        # Convert to DataFrame and sort by importance
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def plot_prediction_probabilities(self, predictions, figsize=(10, 6)):
        """
        Plot the probability distribution of predictions.
        
        Args:
            predictions: List of prediction dictionaries from predict_batch
            figsize: Figure size
        """
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Extract probabilities
        probs_list = []
        for i, pred in enumerate(predictions):
            probs = list(pred['probabilities'].values())
            probs_list.append(probs)
        
        # Create DataFrame
        df = pd.DataFrame(probs_list, columns=self.delay_categories)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(df, annot=True, cmap='Blues', fmt='.2f', linewidths=.5)
        plt.title('Delay Prediction Probabilities')
        plt.ylabel('Flight Chain')
        plt.tight_layout()
        plt.savefig('prediction_probabilities.png')
        plt.show()
    
    def save_preprocessor(self, output_path):
        """Save the preprocessor to disk."""
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        print(f"Preprocessor saved to {output_path}")
    
    def save_model_config(self, output_path):
        """Save the model configuration to disk."""
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Model configuration saved to {output_path}")


def predict_from_csv(model_path, data_path, output_path=None, chain_length=3):
    """
    Make predictions on flight data from a CSV file.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the CSV file containing flight data
        output_path: Path to save predictions (if None, will use data_path + '_predictions.csv')
        chain_length: Number of consecutive flights in a chain
    """
    # Load the predictor
    predictor = FlightDelayPredictor(model_path)
    
    # Load flight data
    df = pd.read_csv(data_path)
    
    # Ensure required columns exist
    required_columns = predictor.preprocessor.categorical_features + \
                      predictor.preprocessor.numerical_features + \
                      predictor.preprocessor.temporal_features
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataset: {missing_columns}")
    
    # Group flights by tail number and sort by schedule time
    flight_chains = []
    for tail_number, group in df.groupby('Tail_Number'):
        sorted_flights = group.sort_values('Schedule_DateTime')
        
        # Create chains of consecutive flights
        for i in range(len(sorted_flights) - chain_length + 1):
            chain = sorted_flights.iloc[i:i+chain_length]
            
            # Check if these flights are consecutive (same day or reasonable gap)
            is_valid_chain = True
            for j in range(1, len(chain)):
                prev_time = pd.to_datetime(chain.iloc[j-1]['Schedule_DateTime']) + \
                            pd.Timedelta(minutes=chain.iloc[j-1]['Flight_Duration_Minutes'])
                curr_time = pd.to_datetime(chain.iloc[j]['Schedule_DateTime'])
                time_diff = (curr_time - prev_time).total_seconds() / 3600  # hours
                
                # If gap is too large (e.g., > 24 hours), this isn't a valid chain
                if time_diff > 24:
                    is_valid_chain = False
                    break
            
            if is_valid_chain:
                flight_chains.append(chain.to_dict('records'))
    
    print(f"Created {len(flight_chains)} flight chains for prediction")
    
    # Make predictions in batches
    batch_size = 64
    all_predictions = []
    
    for i in tqdm(range(0, len(flight_chains), batch_size)):
        batch = flight_chains[i:i+batch_size]
        predictions = predictor.predict_batch(batch)
        all_predictions.extend(predictions)
    
    # Create DataFrame with predictions
    results = []
    for i, chain in enumerate(flight_chains):
        # Extract key information about the flight chain
        last_flight = chain[-1]
        
        result = {
            'Tail_Number': last_flight['Tail_Number'],
            'Origin': last_flight['Origin'],
            'Dest': last_flight['Dest'],
            'Schedule_DateTime': last_flight['Schedule_DateTime'],
            'Predicted_Delay_Level': all_predictions[i]['delay_level'],
            'Predicted_Delay_Category': all_predictions[i]['delay_category']
        }
        
        # Add probabilities for each category
        for cat, prob in all_predictions[i]['probabilities'].items():
            result[f'Prob_{cat.replace(" ", "_")}'] = prob
        
        # Add actual delay if available
        if 'Flight_Delay' in last_flight:
            result['Actual_Delay'] = last_flight['Flight_Delay']
            
            # Add actual delay category
            delay = last_flight['Flight_Delay']
            if delay <= 0:
                actual_category = 'On Time'
            elif delay <= 15:
                actual_category = 'Slight Delay (1-15 min)'
            elif delay <= 30:
                actual_category = 'Minor Delay (16-30 min)'
            elif delay <= 60:
                actual_category = 'Moderate Delay (31-60 min)'
            else:
                actual_category = 'Severe Delay (>60 min)'
            
            result['Actual_Delay_Category'] = actual_category
        
        results.append(result)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    
    if output_path is None:
        output_path = os.path.splitext(data_path)[0] + '_predictions.csv'
    
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Calculate metrics if actual delays are available
    if 'Actual_Delay_Category' in results_df.columns:
        correct_predictions = (results_df['Predicted_Delay_Category'] == results_df['Actual_Delay_Category']).sum()
        accuracy = correct_predictions / len(results_df)
        print(f"Prediction Accuracy: {accuracy:.4f}")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            results_df['Actual_Delay_Category'].map({cat: i for i, cat in enumerate(predictor.delay_categories)}),
            results_df['Predicted_Delay_Level']
        )
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=predictor.delay_categories,
                   yticklabels=predictor.delay_categories)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix_predictions.png')
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Flight delay prediction using SimAM-CNN-MogrifierLSTM model')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                      help='Path to the trained model')
    parser.add_argument('--data', type=str, default='processedDataTest/test_set.csv',
                      help='Path to the CSV file containing flight data')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save predictions')
    parser.add_argument('--chain-length', type=int, default=3,
                      help='Number of consecutive flights in a chain')
    
    args = parser.parse_args()
    
    # Make predictions
    predict_from_csv(args.model, args.data, args.output, args.chain_length)

if __name__ == "__main__":
    main()
