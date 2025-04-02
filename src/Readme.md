# SkyDenied

## Installation Guide

### Setting Up the Environment (cleanData)

```bash
# Install uv from https://github.com/astral-sh/uv

# Create a virtual environment
uv venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install necessary packages
uv pip install pandas numpy scikit-learn

# Run your Python scripts
uv run python cleanData.py
# Or
uv run python cleanRealTime.py
```

### Setting Up the Environment ML

```bash
# Create a new conda environment with Python 3.9 and TensorFlow 2.12
conda create -n tf2.12 -c conda-forge python=3.9 tensorflow=2.12

# Activate the environment
conda activate tf2.12

# Install required packages
conda install -c conda-forge scikit-learn joblib numpy pandas
```

### Verify GPU Detection

After installation, verify that TensorFlow can detect your GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output if successful:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Environment Management Commands

```bash
# Deactivate the environment
conda deactivate

# List all conda environments
conda env list

# Remove an environment
conda remove --name <env_name> --all
```

## Project Structure

- `cleanData.py` - Data preprocessing
- `cleanRealTime.py` - Real-time data cleaning
- `coralLoss.py` - Loss function implementation
- `gruDelayPred.py` - GRU-based delay prediction model
- `lstmDelayPred.py` - LSTM-based delay prediction model

## Important Note

Always install TensorFlow using the conda-forge channel with the exact versions specified above to ensure proper GPU recognition and compatibility.
