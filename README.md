# Neural Network Shape Classifier

A lightweight, real-time shape classifier that uses a custom neural network to identify hand-drawn shapes.

![Shape Classifier Demo](/placeholder.svg?height=300&width=500)

## Overview

This application allows users to draw shapes on a canvas and uses a neural network to classify them in real-time as squares, triangles, or other shapes. The neural network is trained from scratch without using pre-trained models or complex libraries.

## Features

- Real-time shape classification
- Interactive drawing canvas
- Neural network visualization
- Optimized for performance
- Lightweight implementation

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- SciPy
- PyGame

### Setup

1. Clone the repository: `git clone [https://github.com/abdulaziz-python/neural-network](https://github.com/abdulaziz-python/neural-network)`
`cd neural-network`



2. Install dependencies: `pip install numpy scipy pygame`


3. Run the application: `python ai.py`


## Usage

1. Draw a shape (square, triangle, or any other shape) on the canvas using your mouse
2. The neural network will classify the shape in real-time
3. The prediction bars on the left show the confidence level for each class
4. Press 'C' to clear the canvas and draw a new shape
5. Close the window to exit the application

## Technical Details

### Neural Network Architecture

- Input layer: 256 neurons (16x16 grid features)
- Hidden layers: 128 and 64 neurons with Leaky ReLU activation
- Output layer: 3 neurons with Softmax activation (Square, Triangle, Other)
- Optimization: Adam optimizer with Xavier/Glorot initialization

### Feature Extraction

The application extracts features from the drawn shape using:
- Grid-based feature extraction (16x16 grid)
- Gaussian smoothing for noise reduction

### Training Data

The network is trained on procedurally generated examples of:
- Squares (including rotated squares)
- Triangles (various types and orientations)
- Other shapes (circles and polygons)

## Performance Optimizations

- Vectorized mathematical operations
- Efficient memory usage with pre-allocated arrays
- Simplified feature extraction
- Optimized rendering for visualization

## License

This project is licensed under the Apache License - see the LICENSE file for details.


