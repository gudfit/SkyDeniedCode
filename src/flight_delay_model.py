import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ================ DATA PREPROCESSING MODULE ================

class FlightDataPreprocessor:
    """
    Handles preprocessing of flight data for delay prediction model.
    """
    def __init__(self, categorical_features, numerical_features, temporal_features, embedding_dims=None):
        """
        Initialize the preprocessor with feature definitions.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            temporal_features: List of temporal feature names
            embedding_dims: Dictionary mapping categorical features to embedding dimensions
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.temporal_features = temporal_features
        self.embedding_dims = embedding_dims or {}
        
        self.categorical_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.feature_dims = {}
        self.total_dim = 0
        
    def fit(self, df):
        """
        Fit preprocessors on the training data.
        
        Args:
            df: DataFrame containing the training data
        """
        # Fit encoders for categorical features
        for feature in self.categorical_features:
            unique_values = df[feature].unique()
            self.feature_dims[feature] = len(unique_values)
            # Handle both older and newer scikit-learn versions
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                # Fall back to older parameter name for scikit-learn < 1.2
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(df[feature].values.reshape(-1, 1))
            self.categorical_encoders[feature] = encoder
        
        # Fit scaler for numerical features
        if self.numerical_features:
            self.numerical_scaler.fit(df[self.numerical_features])
            
        # Calculate total feature dimension after preprocessing
        self.calculate_total_dim()
        
        return self
    
    
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
        print(f"Calculated feature dimension: {total}")
        return total
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
    def transform_flight_chain(self, flight_chain):
        """
        Transform a sequence of flights into a matrix of features.
        
        Args:
            flight_chain: List of dicts or DataFrame rows containing flight chain data
            
        Returns:
            Matrix of shape (T, D) where T is chain length and D is feature dimension
        """
        T = len(flight_chain)
        X = np.zeros((T, self.total_dim))
        
        for t, flight in enumerate(flight_chain):
            X[t] = self.transform_single_flight(flight)
            
        return X
    
    def calculate_ground_time(self, flight_chain):
        """
        Calculate ground time between consecutive flights.
        
        Args:
            flight_chain: List of flight data points
            
        Returns:
            List of ground times in minutes
        """
        ground_times = []
        
        for i in range(1, len(flight_chain)):
            prev_flight = flight_chain[i-1]
            curr_flight = flight_chain[i]
            
            prev_arr_time = pd.to_datetime(prev_flight['Schedule_DateTime']) + pd.Timedelta(minutes=prev_flight['Flight_Duration_Minutes'])
            curr_dep_time = pd.to_datetime(curr_flight['Schedule_DateTime'])
            
            # Ground time in minutes
            ground_time = (curr_dep_time - prev_arr_time).total_seconds() / 60
            ground_times.append(max(0, ground_time))  # Ensure non-negative
            
        return ground_times


# ================ FLIGHT CHAIN DATASET ================

class FlightChainDataset(Dataset):
    """
    Dataset for flight chains for PyTorch training.
    """
    def __init__(self, data_path, preprocessor, chain_length=3, is_train=True):
        """
        Args:
            data_path: Path to the CSV file containing flight data
            preprocessor: Initialized FlightDataPreprocessor
            chain_length: Number of consecutive flights in a chain (T)
            is_train: If True, will fit the preprocessor; otherwise just transform
        """
        self.data_path = data_path
        self.preprocessor = preprocessor
        self.chain_length = chain_length
        
        # Load data
        self.df = pd.read_csv(data_path)
        
        # Group flights by tail number to form chains
        self.flight_chains = []
        self.delay_labels = []
        
        # Group by tail number and sort by schedule time
        for tail_number, group in self.df.groupby('Tail_Number'):
            sorted_flights = group.sort_values('Schedule_DateTime')
            
            # Create chains of consecutive flights
            for i in range(len(sorted_flights) - chain_length + 1):
                chain = sorted_flights.iloc[i:i+chain_length]
                
                # Check if these flights are consecutive (same day or reasonable gap)
                is_valid_chain = True
                for j in range(1, len(chain)):
                    prev_time = pd.to_datetime(chain.iloc[j-1]['Schedule_DateTime']) + pd.Timedelta(minutes=chain.iloc[j-1]['Flight_Duration_Minutes'])
                    curr_time = pd.to_datetime(chain.iloc[j]['Schedule_DateTime'])
                    time_diff = (curr_time - prev_time).total_seconds() / 3600  # hours
                    
                    # If gap is too large (e.g., > 24 hours), this isn't a valid chain
                    if time_diff > 24:
                        is_valid_chain = False
                        break
                
                if is_valid_chain:
                    # Extract delay label for the last flight (T-th flight)
                    last_flight = chain.iloc[-1]
                    delay_level = self.get_delay_level(last_flight['Flight_Delay'])
                    
                    self.flight_chains.append(chain.to_dict('records'))
                    self.delay_labels.append(delay_level)
        
        # Fit preprocessor if in training mode
        if is_train:
            self.preprocessor.fit(self.df)
    
    def get_delay_level(self, delay_minutes):
        """
        Convert delay in minutes to categorical delay level.
        
        Args:
            delay_minutes: Flight delay in minutes
            
        Returns:
            Integer delay level (0-4)
        """
        if delay_minutes <= 0:
            return 0  # On time or early
        elif delay_minutes <= 15:
            return 1  # Slight delay
        elif delay_minutes <= 30:
            return 2  # Minor delay
        elif delay_minutes <= 60:
            return 3  # Moderate delay
        else:
            return 4  # Severe delay
    
    def __len__(self):
        return len(self.flight_chains)
    
    def __getitem__(self, idx):
        chain = self.flight_chains[idx]
        label = self.delay_labels[idx]
        
        # Transform the flight chain to matrix X
        X = self.preprocessor.transform_flight_chain(chain)
        
        return torch.FloatTensor(X), torch.LongTensor([label])


# ================ MODEL COMPONENTS ================

class SimAM(nn.Module):
    """
    SimAM Attention Module as described in the paper.
    """
    def __init__(self, lambda_val=0.1):
        super(SimAM, self).__init__()
        self.lambda_val = lambda_val
        
    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()
        
        # Clone tensor for computation
        y = x.clone()
        
        # Apply attention channel-wise
        for batch in range(batch_size):
            for channel in range(channels):
                feature_map = x[batch, channel]  # [height, width]
                
                # For each position, compute attention using mean/var of other positions
                for i in range(height):
                    for j in range(width):
                        # Current value
                        t = feature_map[i, j].item()
                        
                        # Create mask to exclude current position
                        mask = torch.ones_like(feature_map)
                        mask[i, j] = 0
                        
                        # Calculate mean and variance of other positions
                        other_values = feature_map[mask.bool()]
                        mu_hat = other_values.mean()
                        sigma_hat_sq = other_values.var()
                        
                        # Calculate energy (Eq. 19)
                        energy = 4 * (sigma_hat_sq + self.lambda_val) / ((t - mu_hat)**2 + 2 * sigma_hat_sq + 2 * self.lambda_val)
                        
                        # Apply attention weight (Eq. 20)
                        y[batch, channel, i, j] = torch.sigmoid(1 / energy) * x[batch, channel, i, j]
        
        return y


class CNNBlock(nn.Module):
    """
    CNN Block with SimAM attention.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=(1, 3), use_simam=True, lambda_val=0.1):
        super(CNNBlock, self).__init__()
        self.use_simam = use_simam
        
        # Convolutional layers (1x1 followed by 3x3 as per Figure 5)
        self.conv_layers = nn.ModuleList()
        prev_channels = in_channels
        
        for k_size in kernel_sizes:
            padding = (k_size - 1) // 2  # Same padding
            conv_layer = nn.Sequential(
                nn.Conv2d(prev_channels, out_channels, kernel_size=k_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.conv_layers.append(conv_layer)
            prev_channels = out_channels
        
        # SimAM attention module
        if use_simam:
            self.simam = SimAM(lambda_val=lambda_val)
    
    def forward(self, x):
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Apply SimAM attention if specified
        if self.use_simam:
            x = self.simam(x)
        
        return x


class MogrifierLSTMCell(nn.Module):
    """
    MogrifierLSTM Cell as described in the paper.
    """
    def __init__(self, input_size, hidden_size, num_rounds=6):
        super(MogrifierLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rounds = num_rounds
        
        # Mogrifier matrices for x (odd rounds) and h (even rounds)
        self.Q_matrices = nn.ModuleList([
            nn.Linear(hidden_size, input_size) 
            for _ in range((num_rounds + 1) // 2)
        ])
        
        self.R_matrices = nn.ModuleList([
            nn.Linear(input_size, hidden_size) 
            for _ in range(num_rounds // 2)
        ])
        
        # LSTM gate parameters
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
    
    def mogrify(self, x, h):
        """
        Apply Mogrifier interaction between input and hidden state.
        
        Args:
            x: Input tensor [batch_size, input_size]
            h: Hidden state tensor [batch_size, hidden_size]
            
        Returns:
            Modified x and h tensors
        """
        x_mod, h_mod = x, h
        
        for i in range(1, self.num_rounds + 1):
            if i % 2 == 1:  # Odd rounds: update x
                idx = i // 2
                x_mod = 2 * torch.sigmoid(self.Q_matrices[idx](h_mod)) * x_mod
            else:  # Even rounds: update h
                idx = (i // 2) - 1
                h_mod = 2 * torch.sigmoid(self.R_matrices[idx](x_mod)) * h_mod
        
        return x_mod, h_mod
    
    def forward(self, x, states):
        h, c = states
        
        # Apply Mogrifier interaction
        x_mod, h_mod = self.mogrify(x, h)
        
        # Apply standard LSTM cell with modified inputs
        h_next, c_next = self.lstm_cell(x_mod, (h_mod, c))
        
        return h_next, c_next


class SimAMCNNMogrifierLSTM(nn.Module):
    """
    Combined SimAM-CNN-MogrifierLSTM model for flight delay prediction.
    """
    def __init__(self, input_size, hidden_size, num_classes=5, 
                 cnn_channels=[32, 64], kernel_sizes=[1, 3], 
                 use_simam=True, lambda_val=0.1, num_rounds=6):
        super(SimAMCNNMogrifierLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # CNN blocks with SimAM attention
        self.cnn_blocks = nn.ModuleList()
        prev_channels = 1  # Treat input as single-channel initially
        
        for out_channels in cnn_channels:
            block = CNNBlock(prev_channels, out_channels, kernel_sizes, use_simam, lambda_val)
            self.cnn_blocks.append(block)
            prev_channels = out_channels
        
        # Calculate CNN output size (depends on input size and CNN architecture)
        # For simplicity, assuming no dimension reduction from CNN
        self.cnn_output_size = cnn_channels[-1] * input_size
        
        # MogrifierLSTM layer
        self.mogrifier_lstm = MogrifierLSTMCell(self.cnn_output_size, hidden_size, num_rounds)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, T, D]
                where T is the chain length and D is feature dimension
        """
        batch_size, T, D = x.size()
        
        # Reshape input for CNN: [batch_size, channels=1, T, D]
        x_cnn = x.unsqueeze(1)
        
        # Apply CNN blocks
        for block in self.cnn_blocks:
            x_cnn = block(x_cnn)
        
        # Reshape for LSTM: [batch_size, T, cnn_output_size]
        _, C, T_out, D_out = x_cnn.size()
        x_lstm = x_cnn.view(batch_size, T_out, C * D_out)
        
        # Initialize LSTM states
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Process sequence with MogrifierLSTM
        for t in range(T_out):
            h, c = self.mogrifier_lstm(x_lstm[:, t], (h, c))
        
        # Final prediction using the last hidden state
        out = self.fc_out(h)
        
        return out


# ================ TRAINING AND EVALUATION ================

def train_model(model, train_loader, val_loader, 
                criterion=nn.CrossEntropyLoss(), 
                optimizer=None, 
                num_epochs=50, 
                device='cuda' if torch.cuda.is_available() else 'cpu',
                model_save_path='best_model.pth'):
    """
    Train the SimAM-CNN-MogrifierLSTM model.
    
    Args:
        model: Initialized model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        model_save_path: Path to save the best model
    
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Default optimizer if none provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Best validation accuracy for model saving
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device).squeeze()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved new best model with validation accuracy: {best_val_acc:.4f}')
    
    return model, history


def plot_training_history(history):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing training history
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        Test accuracy and confusion matrix
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, cm


def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations to each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()


# ================ MAIN EXECUTION ================

if __name__ == "__main__":
    # Define feature sets
    categorical_features = ['Carrier_Airline', 'Tail_Number', 'Origin', 'Dest', 'Orientation']
    numerical_features = ['Flight_Duration_Minutes', 'FTD', 'PFD', 'Flight_Delay']
    temporal_features = ['Schedule_DateTime']
    
    # Define embedding dimensions for categorical features with many levels
    embedding_dims = {
        'Tail_Number': 16,
        'Origin': 8,
        'Dest': 8
    }
    
    # Initialize preprocessor
    preprocessor = FlightDataPreprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        temporal_features=temporal_features,
        embedding_dims=embedding_dims
    )
    
    # Create datasets
    train_dataset = FlightChainDataset(
        data_path='processedDataTest/train_set.csv',
        preprocessor=preprocessor,
        chain_length=3,
        is_train=True
    )
    
    val_dataset = FlightChainDataset(
        data_path='processedDataTest/validation_set.csv',
        preprocessor=preprocessor,
        chain_length=3,
        is_train=False
    )
    
    test_dataset = FlightChainDataset(
        data_path='processedDataTest/test_set.csv',
        preprocessor=preprocessor,
        chain_length=3,
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize model
    input_size = preprocessor.total_dim
    hidden_size = 128
    num_classes = 5  # Delay levels (0-4)
    
    model = SimAMCNNMogrifierLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        cnn_channels=[32, 64],
        kernel_sizes=[1, 3],
        use_simam=True,
        lambda_val=0.1,
        num_rounds=6
    )
    
    # Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=30,
        model_save_path='best_model.pth'
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on test set
    test_acc, cm = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Plot confusion matrix
    class_names = ['On Time', 'Slight Delay', 'Minor Delay', 'Moderate Delay', 'Severe Delay']
    plot_confusion_matrix(cm, class_names)
    
    # Example of using the model for prediction
    def predict_delay(model, flight_chain, preprocessor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Predict delay level for a flight chain.
        
        Args:
            model: Trained model
            flight_chain: List of dicts containing flight data
            preprocessor: Fitted preprocessor
            device: Device to use for prediction
            
        Returns:
            Predicted delay level and class probabilities
        """
        model = model.to(device)
        model.eval()
        
        # Preprocess flight chain
        X = preprocessor.transform_flight_chain(flight_chain)
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        delay_level = predicted.item()
        probs = probabilities.cpu().numpy()[0]
        
        return delay_level, probs
    
    # Usage example (would be run with actual flight data)
    # sample_chain = [flight1_data, flight2_data, flight3_data]
    # delay_level, probs = predict_delay(model, sample_chain, preprocessor)
    # print(f'Predicted Delay Level: {class_names[delay_level]}')
    # print(f'Probabilities: {probs}')
