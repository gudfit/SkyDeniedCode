# SimAM-CNN-MogrifierLSTM Flight Delay Prediction (Simplified)

This project implements a neural network model for predicting flight delay categories for the final flight in a sequence of flights performed by the same aircraft. It utilizes components inspired by the SimAM-CNN-MogrifierLSTM architecture.

**Note:** This implementation represents a simplified version adapted for specific hardware constraints compared to potentially more complex research implementations.

## Model Architecture

The model combines several components:

1.  **CNN Layers with optional SimAM Attention:** `Conv2d` layers process the input sequence (viewed as an "image" where height=time steps, width=features). SimAM attention can be applied after convolution to refine feature maps by focusing on important spatial information without adding parameters.
2.  **Mogrifier LSTM:** An enhanced LSTM cell (`MogrifierLSTMCell`) where the input (`x`) and previous hidden state (`h`) interact and modulate each other over several rounds before being fed into the standard LSTM computation. This aims to improve the cell's ability to capture complex dependencies.
3.  **Output Layer:** A fully connected layer predicts the probability distribution over the defined delay categories for the *last* flight in the input sequence (the T-th flight, typically the 3rd).

## Key Components

*   **SimAM Attention:** A parameter-free attention mechanism used within the CNN blocks to help the network focus on salient features across the time steps and feature dimensions.
*   **Mogrifier LSTM Cell:** Implements the Mogrifier mechanism to enhance the standard LSTM's input/state interactions before calculating the updated cell and hidden states.

## Data Processing (`FlightDataPreprocessor` & `FlightChainDataset`)

The data processing pipeline within `flight_delay_model.py` involves:

1.  **Loading Pre-processed Data:** Assumes input CSVs (`train_set.csv`, etc.) are already generated, containing sequences or identifiable chains.
2.  **Feature Definition:** Specifies which columns are categorical, numerical, and temporal.
3.  **Preprocessor Fitting (on Train data):**
    *   Learns One-Hot Encoding for categorical features (handling unknown values and NaNs).
    *   Fits a StandardScaler for numerical features (handling NaNs via mean imputation).
    *   Defines cyclical (sin/cos) encoding for temporal features (`Schedule_DateTime`).
    *   Saves the fitted preprocessor (`fitted_preprocessor.joblib`).
4.  **Dataset Creation (`FlightChainDataset`):**
    *   Groups flights by `Tail_Number`.
    *   Creates sequences (chains) of a specified `chain_length`.
    *   Uses the *fitted* preprocessor to transform each flight in a chain into a feature vector.
    *   Determines the target delay category for the *last* flight in each chain.
    *   Provides sequences (as tensors `[T, D]`) and labels for the DataLoader.

## Feature Processing

*   **Categorical Features:** `Carrier_Airline`, `Origin`, `Dest`, `Orientation` - Processed using One-Hot Encoding. Missing values treated as a separate category.
*   **Numerical Features:** `Flight_Duration_Minutes`, `FTD`, `PFD` - Normalized using StandardScaler. Missing values imputed with the mean during fitting/transforming.
*   **Temporal Features:** `Schedule_DateTime` - Converted to cyclical features using sine/cosine encoding for hour, day of week, and month.

## Setup and Usage

### Requirements

*   Python 3.x
*   PyTorch
*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   tqdm
*   joblib

```bash
pip install torch pandas numpy scikit-learn matplotlib tqdm joblib
```

## Delay Categories

The model predicts flight delays classified into 5 categories based on the arrival delay of the last flight in the sequence:

*   **0:** On Time/Early (`<= 0` minutes)
*   **1:** Slight Delay (`1` to `15` minutes)
*   **2:** Moderate Delay (`16` to `45` minutes)
*   **3:** Significant Delay (`46` to `120` minutes)
*   **4:** Severe Delay (`> 120` minutes)

## Model Parameters (Defaults in `flight_delay_model.py`)

*   **Input Feature Dimension:** Determined by the `FlightDataPreprocessor` after fitting.
*   **Chain Length (T):** 3
*   **LSTM Hidden Size:** 128
*   **CNN Channels:** `[32, 64]`
*   **CNN Kernel Sizes:** `[(1, 3), (3, 3)]` (List of 2D tuples: `(kernel_height, kernel_width)`)
*   **Use SimAM:** True
*   **Mogrifier Rounds:** 5
*   **Output Classes:** 5 (corresponding to the delay categories)
*   **Batch Size:** 32
*   **Epochs:** 10 (configurable, increase for full training)
*   **Learning Rate:** 0.001
*   **Early Stopping Patience:** 5

## Tutorial

```bash
    python data_processing.py
```

```bash
    python adapt_data.py
```

```bash
    python main.py
```

### Expectation

This script will:

1.  Load the processed data from the `data/` directory.
2.  Fit the `FlightDataPreprocessor` on `train_set.csv` if `fitted_preprocessor.joblib` doesn't exist, or load the existing one.
3.  Create PyTorch Datasets and DataLoaders.
4.  Initialize the `SimAMCNNMogrifierLSTM` model.
5.  Train the model using the training data and validate using the validation data.
6.  Implement early stopping based on validation loss.
7.  Save the best performing model weights to `models/best_simam_cnn_mogrifier_lstm.pth`.
8.  Save the fitted preprocessor state to `models/fitted_preprocessor.joblib`.
9.  Evaluate the best model on the test set.
10. Generate training history and confusion matrix plots in the `plots/` directory.
11. Run a prediction on a sample from the test set.

