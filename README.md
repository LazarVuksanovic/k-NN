# Iris Flower Classification with k-Nearest Neighbors (k-NN)

This project involves using the k-Nearest Neighbors (k-NN) algorithm to classify iris flowers based on their features. Three Python scripts (`3a.py`, `3b.py`, `3c.py`) are provided for different aspects of the classification task.

## Scripts Overview

- **`3a.py`**: Performs k-NN (k=3) classification using the first two features of the iris dataset.
  - Splits data into training and test sets.
  - Evaluates accuracy on the test set.
  - Visualizes training data with colored classes and decision boundaries.

- **`3b.py`**: Explores k-NN performance with different k values (1-15) using the first two features.
  - Plots accuracy on the test set for each k value.
  - Identifies the optimal k based on accuracy.

- **`3c.py`**: Extends k-NN analysis to use all iris dataset features.
  - Compares performance using all features versus only the first two.
  - Provides insights into feature impact on classification accuracy.

## Usage

1. **Setup**:
   - Ensure Python (3.x) and required libraries (`pandas`, `numpy`, `matplotlib`, `sklearn`) are installed.

2. **Dataset**:
   - Use `iris.csv` for iris flower data (features: sepal length, sepal width, petal length, petal width, species).

3. **Execution**:
   - Run scripts (`3a.py`, `3b.py`, `3c.py`) to explore different aspects of iris classification using k-NN.

## Results and Visualizations

- **`3a.py`**:
  - Displays accuracy and decision boundaries for k-NN (k=3) using two features.
  
- **`3b.py`**:
  - Investigates optimal k value for k-NN using two features.
  - Visualizes accuracy trends across different k values.

- **`3c.py`**:
  - Compares k-NN performance using all features versus a subset.
  - Analyzes feature impact on classification accuracy.
