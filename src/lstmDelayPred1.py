# lstmModel_v2.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, concatenate,
    TimeDistributed, Masking
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical # Not needed for CORAL target
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    mean_absolute_error, accuracy_score # More specific imports
)
import matplotlib.pyplot as plt
import joblib

# Import CORAL loss functions
try:
    from coralLoss import coral_loss, levels_from_logits
except ImportError:
    print("Error: coralLoss.py not found. Please ensure it's in the Python path.")
    exit()


# --- Configuration ---
# 1. Data Path (Update this to the file WITH queue features)
DATA_FILE = "../processedDataQUpdated/processed_flights_with_queue_features.csv"

# 2. Output Directory
OUTPUT_DIR = "../lstm_ordinal_model_output_v2" # New output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Ordinal Classification Bins (5 Levels)
DELAY_BINS = [-np.inf, -0.001, 15, 30, 60, np.inf]
DELAY_LABELS = [0, 1, 2, 3, 4] # 0: Early, 1: On Time, 2: Late, 3: Very Late, 4: Severely Late
NUM_CLASSES = len(DELAY_LABELS)
CLASS_NAMES = [f'Level {i}' for i in DELAY_LABELS] # For plotting

# 4. Feature Selection (Including Queue Features)
CATEGORICAL_FEATURES = ['Carrier_Airline', 'Origin', 'Dest', 'Orientation', 'Season']
NUMERICAL_FEATURES = [
    # Original features
    'Flight_Duration_Minutes', 'InterEventTimeMinutes', 'PFD',
    'Departures_In_Window_Origin', 'Arrivals_In_Window_Dest', # From congestion script
    # Decomposed temporal features
    'Hour', 'DayOfWeek', 'Month',
    # Queue Theory Features (Add these)
    'Avg_FTD_min', 'Avg_Flight_Duration_min', 'Arrival_Rate_lambda', 'Service_Rate_mu',
    'Stable_Queue' # Will treat as 0/1 numerical
]
TEMPORAL_FEATURE = 'Schedule_DateTime'

# 5. Sequence and Model Parameters
SEQUENCE_LENGTH = 10
LSTM_UNITS = 64
EMBEDDING_DIM = 10
BATCH_SIZE = 128
EPOCHS = 50
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# --- Helper Functions --- (Keep create_sequences and build_lstm_ordinal_model as before)
def create_sequences(df, sequence_length, numerical_cols, categorical_cols, target_col, scalers, encoders):
    """Creates sequences grouped by Tail_Number."""
    print(f"Creating sequences with length {sequence_length}...")
    all_sequences_num = []
    all_sequences_cat = {cat_col: [] for cat_col in categorical_cols}
    all_targets = []

    df_sorted = df.sort_values(by=['Tail_Number', 'Schedule_DateTime'])

    # Pre-scale numerical data
    # Important: Ensure columns exist before scaling
    valid_numerical_cols = [col for col in numerical_cols if col in df_sorted.columns]
    if len(valid_numerical_cols) < len(numerical_cols):
        print(f"Warning: Missing numerical columns for scaling: {set(numerical_cols) - set(valid_numerical_cols)}")
    if not valid_numerical_cols:
         print("Error: No numerical columns found to scale.")
         return None, None, None
    df_sorted[valid_numerical_cols] = scalers['numerical'].transform(df_sorted[valid_numerical_cols])

    # Pre-encode categorical data
    encoded_cats = {}
    valid_categorical_cols = [col for col in categorical_cols if col in df_sorted.columns]
    if len(valid_categorical_cols) < len(categorical_cols):
        print(f"Warning: Missing categorical columns for encoding: {set(categorical_cols) - set(valid_categorical_cols)}")

    for cat_col in valid_categorical_cols:
        # Handle unseen labels during transform by mapping them to a default value (e.g., len(classes))
        # Here we rely on the model's embedding layer (mask_zero=True) to handle padding (0),
        # assuming unseen labels won't be encoded as 0 by LabelEncoder.
        # A more robust way handles unseen explicitly (e.g., adding an 'UNK' category).
        encoded_cats[cat_col] = df_sorted[cat_col].apply(lambda x: encoders[cat_col].transform([x])[0] if x in encoders[cat_col].classes_ else -1) # Temp encode unseen as -1
        # Check if LabelEncoder assigned 0 to any valid class, shift if necessary (because 0 is used for padding)
        if 0 in encoded_cats[cat_col].unique() and 0 in encoders[cat_col].transform(encoders[cat_col].classes_):
             print(f"Warning: LabelEncoder assigned 0 to a valid class in '{cat_col}'. Shifting indices by +1.")
             encoded_cats[cat_col] = encoded_cats[cat_col] + 1 # Shift everything up by 1
             # Also update encoder mapping if needed elsewhere, though less critical now
        encoded_cats[cat_col] = encoded_cats[cat_col].replace(-1, 0) # Map unseen (-1) to 0 (padding index)


    grouped = df_sorted.groupby('Tail_Number')
    total_groups = len(grouped)
    processed_groups = 0

    for tail_num, group in grouped:
        processed_groups += 1
        if processed_groups % 1000 == 0: # Adjusted print frequency
            print(f"  Processing group {processed_groups}/{total_groups} (Tail: {tail_num})")

        # Extract features for the group
        num_features = group[valid_numerical_cols].values
        target = group[target_col].values

        cat_features_group = {cat_col: encoded_cats[cat_col][group.index] for cat_col in valid_categorical_cols}

        if len(group) > sequence_length: # Only create sequences if group is long enough
            for i in range(len(group) - sequence_length):
                seq_num = num_features[i : i + sequence_length]
                seq_target = target[i + sequence_length]

                all_sequences_num.append(seq_num)
                for cat_col in valid_categorical_cols:
                    seq_cat = cat_features_group[cat_col][i : i + sequence_length]
                    all_sequences_cat[cat_col].append(seq_cat)
                all_targets.append(seq_target)

    print(f"Finished creating sequences. Found {len(all_targets)} sequences.")
    if not all_targets:
        print("Error: No sequences were created. Check sequence_length and data.")
        return None, None, None

    # Pad sequences
    padded_sequences_num = pad_sequences(all_sequences_num, maxlen=sequence_length, padding='pre', dtype='float32', value=0.0) # Pad with 0.0
    padded_sequences_cat = {}
    for cat_col in valid_categorical_cols:
        # Pad with 0, assuming 0 is reserved for padding/unseen
        padded_sequences_cat[cat_col] = pad_sequences(all_sequences_cat[cat_col], maxlen=sequence_length, padding='pre', dtype='int32', value=0)

    return padded_sequences_num, padded_sequences_cat, np.array(all_targets)

def build_lstm_ordinal_model(seq_length, num_features_numeric, cat_feature_info, lstm_units, num_classes):
    """Builds the LSTM model for ordinal classification using CORAL."""
    print("Building LSTM model...")

    input_numeric = Input(shape=(seq_length, num_features_numeric), name='input_numeric')
    # Masking layer for numerical input (handle padding value 0.0)
    masked_numeric = Masking(mask_value=0.0)(input_numeric)

    input_categoricals = []
    embeddings = []
    # Ensure cat_feature_info has the right structure
    valid_cat_feature_info = [info for info in cat_feature_info if len(info) == 3]

    for cat_col, vocab_size, embed_dim in valid_cat_feature_info:
        input_cat = Input(shape=(seq_length,), name=f'input_{cat_col}')
        input_categoricals.append(input_cat)
        # Embedding layer - use mask_zero=True to handle padding index 0
        embedding = Embedding(input_dim=vocab_size, # Vocab size should account for padding=0
                              output_dim=embed_dim,
                              mask_zero=True, # Tells downstream layers to ignore padding
                              name=f'embed_{cat_col}')(input_cat)
        embeddings.append(embedding)

    if embeddings:
        # If using Masking before concat, ensure compatibility or let Embedding's mask propagate
        concatenated_features = concatenate([masked_numeric] + embeddings, name='concatenate_features')
    else:
        concatenated_features = masked_numeric

    # LSTM layer automatically receives the mask from concatenation/embedding
    lstm_out = LSTM(lstm_units, name='lstm_layer')(concatenated_features)

    # Output layer for CORAL (num_classes - 1 logits)
    output_logits = Dense(num_classes - 1, name='output_logits')(lstm_out)

    model = Model(inputs=[input_numeric] + input_categoricals, outputs=output_logits)
    print("Model built successfully.")
    return model


# --- Main Script ---

# 1. Load Data
print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE, parse_dates=[TEMPORAL_FEATURE])
    print(f"Loaded {len(df)} records.")
    # Basic check for essential columns
    essential_cols = ['Tail_Number', 'Schedule_DateTime', 'Flight_Delay'] + CATEGORICAL_FEATURES + [f for f in NUMERICAL_FEATURES if f not in ['Hour','DayOfWeek','Month']]
    missing_essentials = [col for col in essential_cols if col not in df.columns]
    if missing_essentials:
        print(f"Error: Data file is missing essential columns: {missing_essentials}")
        exit()
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Feature Engineering & Target Creation
print("Performing feature engineering and target creation...")
# Decompose temporal feature
df['Hour'] = df[TEMPORAL_FEATURE].dt.hour
df['DayOfWeek'] = df[TEMPORAL_FEATURE].dt.dayofweek
df['Month'] = df[TEMPORAL_FEATURE].dt.month

# Convert Stable_Queue (Boolean) to 0/1 integer
if 'Stable_Queue' in df.columns:
    df['Stable_Queue'] = df['Stable_Queue'].astype(int)
else:
    print("Warning: 'Stable_Queue' column not found. Excluding it from features.")
    NUMERICAL_FEATURES.remove('Stable_Queue') # Remove if not present


# Create Ordinal Target Variable
df['Delay_Level'] = pd.cut(df['Flight_Delay'], bins=DELAY_BINS, labels=DELAY_LABELS, right=True)

# --- Data Cleaning / Preprocessing before split ---
# Handle potential NaNs in target or features
if df['Delay_Level'].isnull().any():
    print("Warning: NaNs found in Delay_Level after binning. Dropping affected rows.")
    df.dropna(subset=['Delay_Level'], inplace=True)
df['Delay_Level'] = df['Delay_Level'].astype(int)

print("Checking for NaNs/Infs in numerical features before split...")
for col in NUMERICAL_FEATURES:
    if col in df.columns:
        # Replace inf with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isnull().any():
            # Impute with median (more robust to outliers than mean)
            median_val = df[col].median()
            print(f"  Warning: NaNs found in numerical feature '{col}'. Imputing with median ({median_val:.4f}).")
            df[col].fillna(median_val, inplace=True)
    else:
         print(f"  Warning: Numerical feature '{col}' listed but not found in DataFrame.")

# Ensure categorical columns are string type before encoding
for cat_col in CATEGORICAL_FEATURES:
     if cat_col in df.columns:
        df[cat_col] = df[cat_col].astype(str).fillna('Missing') # Fill NaNs with 'Missing' category
     else:
        print(f"  Warning: Categorical feature '{cat_col}' listed but not found in DataFrame.")


print("Delay Level Distribution (after cleaning):")
print(df['Delay_Level'].value_counts().sort_index())

# 3. Data Splitting
print("Splitting data into Train/Validation/Test sets...")
# Ensure stratification column exists and has multiple classes
if 'Delay_Level' in df.columns and df['Delay_Level'].nunique() > 1:
    stratify_col = df['Delay_Level']
    print("Stratifying split by Delay_Level.")
else:
    stratify_col = None
    print("Warning: Cannot stratify split. Delay_Level missing or has only one class.")

try:
    train_val_df, test_df = train_test_split(df, test_size=TEST_SPLIT, random_state=42, stratify=stratify_col)
    # Adjust val_split fraction for the second split
    val_split_adjusted = VAL_SPLIT / (1.0 - TEST_SPLIT) if (1.0 - TEST_SPLIT) > 0 else VAL_SPLIT
    train_df, val_df = train_test_split(train_val_df, test_size=val_split_adjusted, random_state=42, stratify=train_val_df['Delay_Level'] if stratify_col is not None else None)
except ValueError as e:
     print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
     train_val_df, test_df = train_test_split(df, test_size=TEST_SPLIT, random_state=42)
     val_split_adjusted = VAL_SPLIT / (1.0 - TEST_SPLIT) if (1.0 - TEST_SPLIT) > 0 else VAL_SPLIT
     train_df, val_df = train_test_split(train_val_df, test_size=val_split_adjusted, random_state=42)


print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# 4. Preprocessing (Fit on Training Data Only)
print("Setting up preprocessing (fitting scalers/encoders on training data)...")
# Numerical Scaler
# Filter NUMERICAL_FEATURES to only include columns actually present in train_df
active_numerical_features = [col for col in NUMERICAL_FEATURES if col in train_df.columns]
if not active_numerical_features:
    print("Error: No numerical features found in the training data to scale.")
    exit()
scaler_numerical = StandardScaler()
scaler_numerical.fit(train_df[active_numerical_features])
joblib.dump(scaler_numerical, os.path.join(OUTPUT_DIR, 'scaler_numerical.joblib'))
scalers = {'numerical': scaler_numerical}
print(f"  Fitted StandardScaler on: {active_numerical_features}")

# Categorical Encoders
encoders = {}
cat_feature_model_info = []
active_categorical_features = [col for col in CATEGORICAL_FEATURES if col in train_df.columns]
print(f"  Fitting LabelEncoders on: {active_categorical_features}")
for cat_col in active_categorical_features:
    encoder = LabelEncoder()
    # Fit encoder ONLY on training data categories
    encoder.fit(train_df[cat_col]) # Already ensured string and filled NaNs
    # Vocab size = number of classes + 1 (for padding index 0)
    # We will shift labels during sequence creation if 0 is used by the encoder
    vocab_size = len(encoder.classes_) + 1
    encoders[cat_col] = encoder
    joblib.dump(encoder, os.path.join(OUTPUT_DIR, f'encoder_{cat_col}.joblib'))
    cat_feature_model_info.append((cat_col, vocab_size, EMBEDDING_DIM))
    print(f"    Encoder for '{cat_col}': Vocab size = {vocab_size} (includes padding)")


# 5. Create Sequences
# Use the filtered active feature lists
X_train_num, X_train_cat, y_train = create_sequences(train_df, SEQUENCE_LENGTH, active_numerical_features, active_categorical_features, 'Delay_Level', scalers, encoders)
X_val_num, X_val_cat, y_val = create_sequences(val_df, SEQUENCE_LENGTH, active_numerical_features, active_categorical_features, 'Delay_Level', scalers, encoders)
X_test_num, X_test_cat, y_test = create_sequences(test_df, SEQUENCE_LENGTH, active_numerical_features, active_categorical_features, 'Delay_Level', scalers, encoders)

# Check if sequence creation was successful
if y_train is None or y_val is None or y_test is None:
    print("Error: Sequence creation failed. Exiting.")
    exit()
if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
    print("Error: No sequences generated for one or more datasets. Check data and sequence length.")
    exit()


# Prepare model inputs (handle cases with no categorical features)
X_train_inputs = [X_train_num] + [X_train_cat[cat_col] for cat_col, _, _ in cat_feature_model_info if cat_col in X_train_cat]
X_val_inputs = [X_val_num] + [X_val_cat[cat_col] for cat_col, _, _ in cat_feature_model_info if cat_col in X_val_cat]
X_test_inputs = [X_test_num] + [X_test_cat[cat_col] for cat_col, _, _ in cat_feature_model_info if cat_col in X_test_cat]

print(f"Training sequences shape: Num={X_train_num.shape}, Cat={ {k:v.shape for k,v in X_train_cat.items()} }, Target={y_train.shape}")
print(f"Validation sequences shape: Num={X_val_num.shape}, Cat={ {k:v.shape for k,v in X_val_cat.items()} }, Target={y_val.shape}")
print(f"Test sequences shape: Num={X_test_num.shape}, Cat={ {k:v.shape for k,v in X_test_cat.items()} }, Target={y_test.shape}")


# 6. Build and Compile Model
# Pass the actual number of numerical features used
num_active_numeric = X_train_num.shape[-1]
model = build_lstm_ordinal_model(
    seq_length=SEQUENCE_LENGTH,
    num_features_numeric=num_active_numeric,
    cat_feature_info=cat_feature_model_info, # Uses active categorical features
    lstm_units=LSTM_UNITS,
    num_classes=NUM_CLASSES
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=coral_loss(num_classes=NUM_CLASSES))
model.summary()

# 7. Train Model
print("Starting model training...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train_inputs,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_inputs, y_val),
    callbacks=callbacks,
    verbose=1
)

model.save(os.path.join(OUTPUT_DIR, 'lstm_ordinal_model_v2.keras'))
print("Model saved.")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (CORAL)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'))
plt.show()


# 8. Evaluate Model with Enhanced Metrics
print("\n--- Evaluating Model on Test Set ---")
test_loss = model.evaluate(X_test_inputs, y_test, verbose=0)
print(f"Test Loss (CORAL): {test_loss:.4f}")

# Predict levels on the test set
y_pred_logits_test = model.predict(X_test_inputs)
y_pred_levels_test = levels_from_logits(y_pred_logits_test)

# --- Ordinal Classification Metrics ---
# Accuracy (Exact Level Match)
accuracy = accuracy_score(y_test, y_pred_levels_test)
print(f"Test Accuracy (Exact Level Match): {accuracy:.4f}")

# Mean Absolute Error (MAE on Levels) - Measures average level difference
mae = mean_absolute_error(y_test, y_pred_levels_test)
print(f"Test MAE (on Levels): {mae:.4f} (+/- levels)")

# Classification Report (Precision, Recall, F1-score per class)
print("\nClassification Report (on Levels):")
# Use DELAY_LABELS (0-4) and corresponding CLASS_NAMES
report = classification_report(y_test, y_pred_levels_test, labels=DELAY_LABELS, target_names=CLASS_NAMES, zero_division=0)
print(report)
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion Matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_levels_test, labels=DELAY_LABELS)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figure size
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title('Confusion Matrix')
plt.tight_layout() # Adjust layout
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.show()


print("\nEvaluation complete. Results saved in:", OUTPUT_DIR)
print("Script finished.")
