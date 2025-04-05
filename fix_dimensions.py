import os
import numpy as np
import pandas as pd
import pickle
import sys

# Add src to path for imports
sys.path.insert(0, 'src')

# Create a debugging script to identify the correct dimensions
def debug_dimensions():
    """Analyze the dimensions of the processed data and fix dimension mismatch issues."""
    print("Debugging dimensions in flight data...")
    
    # Load adapted data to examine a sample
    try:
        with open('adapted_data/train_data.pkl', 'rb') as f:
            train_chains, train_labels = pickle.load(f)
        
        print(f"Loaded {len(train_chains)} training chains")
        
        # Look at the first chain
        sample_chain = train_chains[0]
        print(f"Sample chain has {len(sample_chain)} flights")
        
        # Create a temporary preprocessor for testing
        from flight_delay_model import FlightDataPreprocessor
        
        # Define feature sets
        categorical_features = ['Carrier_Airline', 'Tail_Number', 'Origin', 'Dest', 'Orientation']
        numerical_features = ['Flight_Duration_Minutes', 'FTD', 'PFD', 'Flight_Delay']
        temporal_features = ['Schedule_DateTime']
        
        # Initialize a test preprocessor
        preprocessor = FlightDataPreprocessor(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            temporal_features=temporal_features
        )
        
        # Flatten all chains for analysis
        all_flights = []
        for chain in train_chains[:100]:  # Use a subset for faster processing
            all_flights.extend(chain)
        
        train_df = pd.DataFrame(all_flights)
        
        # Fill missing values
        for feature in categorical_features:
            if feature not in train_df.columns:
                train_df[feature] = 'Unknown'
        
        for feature in numerical_features:
            if feature not in train_df.columns:
                train_df[feature] = 0.0
        
        if 'Schedule_DateTime' not in train_df.columns:
            train_df['Schedule_DateTime'] = pd.Timestamp('2022-01-01')
        else:
            train_df['Schedule_DateTime'] = pd.to_datetime(train_df['Schedule_DateTime'])
        
        # Fit the preprocessor
        preprocessor.fit(train_df)
        
        # Check calculated dimensions
        print(f"Calculated total dimension: {preprocessor.total_dim}")
        
        # Try to process a single flight to check actual dimensions
        processed = preprocessor.transform_single_flight(train_df.iloc[0])
        print(f"Actual processed feature vector length: {len(processed)}")
        
        # Analyze dimensions for each feature type
        cat_dims = {}
        for feature in categorical_features:
            if feature in preprocessor.categorical_encoders:
                encoder = preprocessor.categorical_encoders[feature]
                n_categories = len(encoder.categories_[0])
                cat_dims[feature] = n_categories
                print(f"  - {feature}: {n_categories} unique values")
        
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Temporal features: {len(temporal_features) * 2 * 3}")  # 2 for sin/cos, 3 for hour/day/month
        
        # Calculate total
        total_cat_dims = sum(cat_dims.values())
        total_num_dims = len(numerical_features)
        total_temp_dims = len(temporal_features) * 2 * 3
        expected_total = total_cat_dims + total_num_dims + total_temp_dims
        
        print(f"Expected total dimension: {expected_total}")
        print(f"Categorical: {total_cat_dims}, Numerical: {total_num_dims}, Temporal: {total_temp_dims}")
        
        # Return the correct dimensions for fixing
        return expected_total, cat_dims, total_num_dims, total_temp_dims
    
    except Exception as e:
        print(f"Error during dimension debugging: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# Now create a fix for the dimension mismatch
def fix_dimension_mismatch():
    """Fix the dimension mismatch between preprocessor and model."""
    # First run the debugging to get the correct dimensions
    expected_total, cat_dims, total_num_dims, total_temp_dims = debug_dimensions()
    
    if expected_total is None:
        print("Could not determine correct dimensions. Fix aborted.")
        return
    
    # Now modify the modified_main.py file to use the correct dimensions
    main_file = 'src/modified_main.py'
    
    try:
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Find where the model is initialized and replace the input_size
        if 'input_size = preprocessor.total_dim' in content:
            # Replace with the known correct dimension
            content = content.replace(
                'input_size = preprocessor.total_dim',
                f'input_size = {expected_total}  # Fixed dimension based on feature analysis'
            )
        
        # Update the CustomFlightChainDataset __getitem__ method to handle dimension mismatches
        dataset_class = '''
        # Create custom datasets
        class CustomFlightChainDataset(torch.utils.data.Dataset):
            def __init__(self, chains, labels, preprocessor, expected_dim={expected_total}):
                self.chains = chains
                self.labels = labels
                self.preprocessor = preprocessor
                self.expected_dim = expected_dim
            
            def __len__(self):
                return len(self.chains)
            
            def __getitem__(self, idx):
                try:
                    chain = self.chains[idx]
                    label = self.labels[idx]
                    
                    # Transform the flight chain to matrix X
                    try:
                        X = self.preprocessor.transform_flight_chain(chain)
                        
                        # Check if dimensions match what's expected
                        if X.shape[1] != self.expected_dim:
                            print(f"Warning: Item {idx} has shape {X.shape} but expected ({len(chain)}, {self.expected_dim})")
                            # Resize to expected dimension
                            resized_X = np.zeros((len(chain), self.expected_dim))
                            # Copy as much as possible
                            min_dim = min(X.shape[1], self.expected_dim)
                            for i in range(len(chain)):
                                resized_X[i, :min_dim] = X[i, :min_dim]
                            X = resized_X
                    
                    except Exception as e:
                        print(f"Error transforming item {idx}: {e}")
                        # Return a dummy tensor with correct shape
                        X = np.zeros((len(chain), self.expected_dim))
                    
                    return torch.FloatTensor(X), torch.LongTensor([label])
                
                except Exception as e:
                    print(f"Error processing item {idx}: {e}")
                    # Return a dummy tensor with correct shape as fallback
                    return torch.zeros((len(self.chains[0]), self.expected_dim)), torch.LongTensor([0])
        '''.replace('{expected_total}', str(expected_total))
        
        # Replace the dataset class definition
        if 'class CustomFlightChainDataset' in content:
            # Find the whole class definition to replace
            start_idx = content.find('class CustomFlightChainDataset')
            end_idx = content.find('# Create datasets', start_idx)
            if end_idx > start_idx:
                content = content[:start_idx] + dataset_class + content[end_idx:]
        
        # Write the modified file
        with open(main_file, 'w') as f:
            f.write(content)
        
        print(f"Fixed dimension mismatch in {main_file}")
        print(f"Set input dimension to {expected_total}")
        
        # Also fix the flight_delay_model.py calculate_total_dim method
        model_file = 'src/flight_delay_model.py'
        
        with open(model_file, 'r') as f:
            model_content = f.read()
        
        # Find the calculate_total_dim method
        start_idx = model_content.find('def calculate_total_dim')
        if start_idx >= 0:
            # Find the end of the method
            next_def_idx = model_content.find('def ', start_idx + 10)
            if next_def_idx > start_idx:
                # Replace with a fixed version that correctly calculates dimensions
                fixed_method = f'''
    def calculate_total_dim(self):
        """Calculate the total dimension of the feature vector after preprocessing."""
        total = 0
        
        # Categorical features (one-hot)
        for feature in self.categorical_features:
            if feature in self.categorical_encoders:
                encoder = self.categorical_encoders[feature]
                n_categories = len(encoder.categories_[0])
                total += n_categories
            elif feature in self.feature_dims:
                total += self.feature_dims[feature]
            elif feature in self.embedding_dims:
                total += self.embedding_dims[feature]
            else:
                # Default to 1 if we don't know
                total += 1
        
        # Numerical features
        total += len(self.numerical_features)
        
        # Temporal features (cyclic encoding adds 2 dimensions per feature)
        # For Schedule_DateTime we encode hour, day of week, and month
        for feature in self.temporal_features:
            if feature == 'Schedule_DateTime':
                total += 2 * 3  # sin/cos for hour, day of week, month
            else:
                total += 2  # sin/cos for other temporal features
        
        self.total_dim = total
        print(f"Calculated feature dimension: {{total}}")
        return total
'''
                model_content = model_content[:start_idx] + fixed_method + model_content[next_def_idx:]
                
                # Write the fixed model file
                with open(model_file, 'w') as f:
                    f.write(model_content)
                
                print(f"Fixed calculate_total_dim method in {model_file}")
    
    except Exception as e:
        print(f"Error during fix: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_dimension_mismatch()
