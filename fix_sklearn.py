import sys
import os

# Create a new version of the flight_delay_model.py file with fixes
original_file = 'src/flight_delay_model.py'
backup_file = 'src/flight_delay_model.py.backup'
fixed_file = 'src/flight_delay_model_fixed.py'

# First, create a backup of the original file
if not os.path.exists(backup_file):
    with open(original_file, 'r') as src, open(backup_file, 'w') as dst:
        dst.write(src.read())
    print(f"Backup created at {backup_file}")

# Now, fix the file
with open(original_file, 'r') as file:
    lines = file.readlines()

with open(fixed_file, 'w') as file:
    in_transform_method = False
    transform_written = False
    
    for i, line in enumerate(lines):
        # Fix the OneHotEncoder initialization for newer scikit-learn versions
        if 'OneHotEncoder' in line and ('sparse=' in line or 'sparse =' in line):
            file.write("            # Handle both older and newer scikit-learn versions\n")
            file.write("            try:\n")
            file.write("                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')\n")
            file.write("            except TypeError:\n")
            file.write("                # Fall back to older parameter name for scikit-learn < 1.2\n")
            file.write("                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')\n")
        
        # Check if we're entering the transform_single_flight method
        elif 'def transform_single_flight(' in line:
            in_transform_method = True
            transform_written = False
            file.write(line)  # Write the method declaration
        
        # If we're in the transform method but haven't rewritten it yet
        elif in_transform_method and not transform_written:
            # If we encounter the next method or class, we've gone past transform_single_flight
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                # Write our fixed transform_single_flight method first
                file.write('''
    def transform_single_flight(self, flight_data):
        """
        Transform a single flight's data into feature vector.
        
        Args:
            flight_data: Series or dict containing a single flight's data
            
        Returns:
            Numpy array of processed features
        """
        features = []
        
        # Process categorical features
        for feature in self.categorical_features:
            if feature not in flight_data:
                # Skip if feature is missing
                continue
                
            value = flight_data[feature]
            # Get one-hot encoding
            if feature in self.categorical_encoders:
                encoder = self.categorical_encoders[feature]
                one_hot = encoder.transform(np.array([[value]]))
                # Convert sparse matrix to dense array if necessary
                if hasattr(one_hot, 'toarray'):
                    one_hot = one_hot.toarray()
                features.extend(one_hot.flatten())
                
        # Process numerical features
        if self.numerical_features:
            try:
                numerical_values = np.array([flight_data.get(f, 0) for f in self.numerical_features]).reshape(1, -1)
                scaled_values = self.numerical_scaler.transform(numerical_values).flatten()
                features.extend(scaled_values)
            except Exception as e:
                print(f"Error processing numerical features: {e}")
                print(f"Available features: {list(flight_data.keys())}")
                print(f"Looking for: {self.numerical_features}")
                # Use zeros as fallback
                features.extend([0] * len(self.numerical_features))
        
        # Process temporal features (convert to cyclical)
        for feature in self.temporal_features:
            if feature == 'Schedule_DateTime':
                if feature not in flight_data:
                    # Use default date if missing
                    import pandas as pd
                    dt = pd.Timestamp('2022-01-01')
                else:
                    # Convert to datetime if it's a string
                    dt = pd.to_datetime(flight_data[feature])
                
                # Hour of day (0-23) -> cyclical encoding
                hour = dt.hour
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                features.extend([hour_sin, hour_cos])
                
                # Day of week (0-6) -> cyclical encoding
                dow = dt.dayofweek
                dow_sin = np.sin(2 * np.pi * dow / 7)
                dow_cos = np.cos(2 * np.pi * dow / 7)
                features.extend([dow_sin, dow_cos])
                
                # Month (1-12) -> cyclical encoding
                month = dt.month
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)
                features.extend([month_sin, month_cos])
        
        return np.array(features)
'''
                )
                transform_written = True
                file.write(line)  # Write the current line (start of next method)
            # Skip the original transform_single_flight method lines
            elif transform_written or (line.strip() and not line.startswith(' ' * 8) and not line.startswith('\t\t')):
                transform_written = True
                file.write(line)
        else:
            file.write(line)

# Replace the original file with the fixed one
import shutil
shutil.move(fixed_file, original_file)
print(f"Fixed {original_file} to handle different scikit-learn versions and sparse matrices.")
