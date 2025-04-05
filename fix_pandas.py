import os

# Path to the flight_delay_model.py file
file_path = 'src/flight_delay_model.py'

# Read the file content
with open(file_path, 'r') as file:
    content = file.read()

# Ensure pandas is imported at the top of the file
if 'import pandas as pd' not in content:
    # Add the import after other standard imports
    if 'import numpy as np' in content:
        content = content.replace('import numpy as np', 'import numpy as np\nimport pandas as pd')
    else:
        # If numpy isn't there, add it near the top of the file
        content = 'import pandas as pd\n' + content

# Fix the transform_single_flight method to make sure pd is available
if 'def transform_single_flight(' in content:
    # Replace the line that causes the issue with a version that doesn't need pd from outer scope
    content = content.replace(
        "                    import pandas as pd", 
        "                    # pandas should be imported at the file level"
    )

# Also fix the issue with any other pd references
if "cannot access local variable 'pd'" in content:
    # Add import inside any method that might be missing it
    method_beginnings = [
        'def transform_single_flight(',
        'def transform_flight_chain(',
        'def calculate_ground_time('
    ]
    
    for method_start in method_beginnings:
        if method_start in content:
            method_pos = content.find(method_start)
            indent = content[method_pos:].find('    ') + 4
            # Add an import after the method docstring if not already there
            docstring_end = content.find('"""', method_pos + len(method_start) + 10)
            if docstring_end > 0:
                insert_pos = docstring_end + 3
                spaces = ' ' * indent
                import_line = f'\n{spaces}# Ensure pandas is available\n{spaces}import pandas as pd\n'
                
                # Only add if not already there
                if 'import pandas as pd' not in content[method_pos:method_pos + 500]:
                    content = content[:insert_pos] + import_line + content[insert_pos:]

# Write the updated file
with open(file_path, 'w') as file:
    file.write(content)

print(f"Fixed pandas import issues in {file_path}")

# Additionally, let's also create a simpler version of the transform_single_flight method
# that doesn't rely on complex error handling that might cause issues

with open(file_path, 'r') as file:
    content = file.read()

# Find the transform_single_flight method
start_pos = content.find('def transform_single_flight(')
if start_pos >= 0:
    # Find the end of the method
    next_def_pos = content.find('def ', start_pos + 10)
    
    # Replace the method with a simpler version
    simpler_method = '''
    def transform_single_flight(self, flight_data):
        """
        Transform a single flight's data into feature vector.
        
        Args:
            flight_data: Series or dict containing a single flight's data
            
        Returns:
            Numpy array of processed features
        """
        import pandas as pd
        import numpy as np
        
        features = []
        
        # Process categorical features
        for feature in self.categorical_features:
            if feature not in flight_data:
                # Skip missing features
                continue
                
            value = flight_data[feature]
            # Get one-hot encoding
            if feature in self.categorical_encoders:
                encoder = self.categorical_encoders[feature]
                one_hot = encoder.transform(np.array([[value]]))
                # Convert to dense if it's sparse
                if hasattr(one_hot, 'toarray'):
                    one_hot = one_hot.toarray()
                features.extend(one_hot.flatten())
        
        # Process numerical features
        if self.numerical_features:
            # Get values, using 0 for missing features
            numerical_values = []
            for f in self.numerical_features:
                if f in flight_data:
                    numerical_values.append(flight_data[f])
                else:
                    numerical_values.append(0)
            
            numerical_values = np.array(numerical_values).reshape(1, -1)
            scaled_values = self.numerical_scaler.transform(numerical_values).flatten()
            features.extend(scaled_values)
        
        # Process temporal features
        for feature in self.temporal_features:
            if feature == 'Schedule_DateTime':
                # Use default date if feature is missing
                if feature not in flight_data:
                    dt = pd.Timestamp('2022-01-01')
                else:
                    dt = pd.to_datetime(flight_data[feature])
                
                # Hour encoding
                hour = dt.hour
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                features.extend([hour_sin, hour_cos])
                
                # Day of week encoding
                dow = dt.dayofweek
                dow_sin = np.sin(2 * np.pi * dow / 7)
                dow_cos = np.cos(2 * np.pi * dow / 7)
                features.extend([dow_sin, dow_cos])
                
                # Month encoding
                month = dt.month
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)
                features.extend([month_sin, month_cos])
        
        return np.array(features)
'''
    
    if next_def_pos > 0:
        content = content[:start_pos] + simpler_method + content[next_def_pos:]
    else:
        # If we couldn't find the next def, replace to the end of the class
        end_class_pos = content.find('class ', start_pos)
        if end_class_pos > 0:
            content = content[:start_pos] + simpler_method + content[end_class_pos:]
        else:
            # If we can't locate the boundaries reliably, don't make the change
            pass

    # Write the updated file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Replaced transform_single_flight with simpler version in {file_path}")
