# Spam Classification Project

This project demonstrates how to classify emails or messages as **spam** or **not spam** using machine learning techniques. It includes data preprocessing, feature extraction, model training, evaluation, and visualization of results.

---

## Project Workflow

### 1. Import Necessary Libraries
The project uses a variety of Python libraries:
- `pandas` for data manipulation.
- `re` and `string` for text preprocessing.
- `scikit-learn` for machine learning models and metrics.
- `nltk` for natural language processing.
- `joblib` for saving the best-performing model.
- `matplotlib` and `seaborn` for visualizations.

### 2. Load and Preprocess Data
#### **Function: `load_data(file_path)`**
- Loads the dataset from a CSV file.
- Displays a confirmation message once data is successfully loaded.

#### **Function: `preprocess_text(text)`**
- Cleans and tokenizes the text data:
  - Converts text to lowercase.
  - Removes URLs, digits, and punctuation.
  - Removes stopwords using NLTK's predefined list.
  - Returns the cleaned text for each email or message.

#### **Function: `preprocess_data(data)`**
- Applies the `preprocess_text` function to all text entries in the dataset, creating a new column `cleaned_text`.

---

### 3. Feature Extraction
#### **Function: `extract_features(data, max_features=10000)`**
- Converts cleaned text into numerical features using TF-IDF vectorization.
  - TF-IDF measures the importance of words in a document relative to the corpus.
- Returns the feature matrix `X` and the fitted `tfidf_vectorizer`.

---

### 4. Data Splitting
#### **Function: `split_data(X, y, test_size=0.3)`**
- Splits the dataset into training and test sets:
  - `X_train` and `X_test`: Feature matrices for training and testing.
  - `y_train` and `y_test`: Labels for training and testing.
- Default split ratio: 70% training, 30% testing.

---

### 5. Model Training and Evaluation
#### **Function: `evaluate_models(models, X_train, X_test, y_train, y_test)`**
- Trains multiple models and evaluates their performance:
  - **Naive Bayes**: A probabilistic model suitable for text classification.
  - **Logistic Regression**: A linear model for binary classification.
  - **Random Forest**: An ensemble method using multiple decision trees.
- Metrics evaluated:
  - **Accuracy**: Correct predictions / total predictions.
  - **Precision**: Proportion of correctly predicted spam out of all predicted spam.
  - **Recall**: Proportion of correctly predicted spam out of all actual spam.
  - **F1 Score**: Harmonic mean of precision and recall.
- Prints and stores metrics for each model.

---

### 6. Save the Best Model
#### **Function: `save_best_model(results)`**
- Identifies the model with the highest accuracy.
- Saves the best model to `best_spam_classifier.pkl` using `joblib`.
- Returns the name and results of the best model.

---

### 7. Confusion Matrix Visualization
#### **Function: `plot_confusion_matrix(y_test, y_pred, model_name)`**
- Displays a heatmap of the confusion matrix for the best model.
- Visualizes true positives, false positives, true negatives, and false negatives.

---

### 8. Compare Model Performance
#### **Function: `plot_metrics(results)`**
- Creates a bar chart comparing the performance of all models.
- Metrics visualized:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

## How to Use

1. **Prepare the Dataset**:
   - Save your email or message dataset as `combined_data.csv`.
   - Ensure it contains:
     - A `text` column with the message content.
     - A `label` column where `1` indicates spam and `0` indicates not spam.

2. **Run the Code**:
   - Ensure all dependencies are installed.
   - Execute the script to:
     - Preprocess the data.
     - Train multiple models.
     - Evaluate and save the best model.
     - Visualize performance metrics and confusion matrix.

3. **Analyze Results**:
   - The confusion matrix shows the classification accuracy.
   - The performance chart provides a detailed comparison of all models.

---

## Dependencies
- `Python 3.x`
- Libraries:
  - `pandas`, `scikit-learn`, `nltk`, `joblib`
  - `matplotlib`, `seaborn`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Project Structure
- **`spam_classifier.ipynb`**: Contains all functions and main code.
- **`combined_data.csv`**: Dataset for training and testing.
- **`best_spam_classifier.pkl`**: Saved model file for the best-performing model.

---

## License
This project is licensed under the [MIT License](LICENSE).

Feel free to adapt and enhance this spam classifier to suit your needs!
