# Next Word Prediction

A machine learning project that implements next word prediction using deep learning models with LSTM and GRU architectures.

## Overview

This project builds and compares two different neural network models for predicting the next word in a sequence:
- **LSTM (Long Short-Term Memory)** model
- **GRU (Gated Recurrent Unit)** model

The models are trained on text data and can generate predictions for completing sentences or continuing text sequences.

## Features

- Text preprocessing and cleaning
- Tokenization using NLTK
- Sequence padding for uniform input length
- Two neural network architectures (LSTM vs GRU)
- Word prediction functionality
- Automatic text generation

## Requirements

The project requires the following Python packages:

```
tensorflow
numpy
pandas
nltk
```


## Dataset

The project uses a text file (`1661-0.txt`) as training data. The text is:
- Cleaned to remove non-ASCII characters
- Tokenized into sentences and words
- Converted to numerical sequences for model training

## Model Architecture

### LSTM Model
- **Embedding Layer**: 64-dimensional word embeddings
- **LSTM Layer**: 100 units
- **Dense Output Layer**: Softmax activation for vocabulary prediction
- **Loss Function**: Sparse categorical crossentropy
- **Optimizer**: Adam

### GRU Model
- **Embedding Layer**: 64-dimensional word embeddings
- **GRU Layer**: 100 units (alternative to LSTM)
- **Dense Output Layer**: Softmax activation for vocabulary prediction
- **Loss Function**: Sparse categorical crossentropy
- **Optimizer**: Adam

## Usage

### Running the Notebook

1. Open `LTSM-GRU.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially to:
   - Load and preprocess the text data
   - Train both LSTM and GRU models
   - Generate predictions

### Making Predictions

The notebook includes a `predict_word()` function that can predict the next word given a seed text:

```python
# Example usage
seed_text = "My name"
next_word = predict_word(seed_text, model=model)  # Using LSTM model
next_word_gru = predict_word(seed_text, model=modelGRU)  # Using GRU model
```

### Text Generation

The project can generate complete sentences by iteratively predicting the next word:

```python
text = "My name "
while not text[-1]=="." and len(text)<100:
    text += predict_word(text) + " "
    print(text)
```

## Data Preprocessing Steps

1. **Text Cleaning**: Remove non-ASCII characters and normalize whitespace
2. **Tokenization**: Split text into sentences and words using NLTK
3. **Vocabulary Building**: Create word-to-index and index-to-word mappings
4. **Sequence Generation**: Create input sequences of varying lengths
5. **Padding**: Pad sequences to uniform length for batch processing
6. **Train/Target Split**: Separate input sequences from target words

## Training

Both models are trained with:
- **Epochs**: 50 (with additional 10 epochs for fine-tuning)
- **Batch Processing**: Automatic batching by Keras
- **Validation**: Training accuracy monitoring

## Model Comparison

The project allows direct comparison between LSTM and GRU architectures:
- Both models use identical preprocessing and architecture (except for the recurrent layer)
- Training performance can be compared through accuracy metrics
- Text generation quality can be evaluated subjectively

## Potential Improvements

- Add validation split for better model evaluation
- Implement beam search for better text generation
- Add temperature parameter for controlling randomness in predictions
- Include model saving/loading functionality
- Add evaluation metrics (perplexity, BLEU score)
- Experiment with different embedding dimensions and hidden units

