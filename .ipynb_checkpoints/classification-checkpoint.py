import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier



# Set the NLTK data download path
nltk_data_path = 'C:/Users/HP/Desktop/ml/project1 classification/nltk_data'



# Add the path where NLTK data will be downloaded
nltk.data.path.append(nltk_data_path)

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)


# Load the dataset
file_path = 'C:/Users/HP/Desktop/ml/project1 classification/CEAS_08.csv'
df = pd.read_csv(file_path)

df.dropna(inplace=True)
print(df.isnull().sum())
# Define stopwords and lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to extract URLs from text
def extract_urls(text):
    if not isinstance(text, str):
        return []
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls

# Function to clean email text (body, subject)
def clean_text(text):
    if not isinstance(text, str):
        text = ''
    # Remove URLs, special characters, and numbers
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # Tokenize, remove stopwords, and lemmatize
    words = word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply URL extraction and text cleaning
df['extracted_urls'] = df['body'].apply(extract_urls)
df['body'] = df['body'].apply(clean_text)
df['subject'] = df['subject'].apply(clean_text)

# Ensure columns are strings
def column_as_string(X):
    return X.astype(str)

# Features and labels
X = df[['body', 'extracted_urls', 'subject', 'sender']]  # Include sender here
y = df['label']

# Train-test split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessor: handle 'body', 'extracted_urls', and 'subject' columns separately
preprocessor = ColumnTransformer(
    transformers=[
        ('body', Pipeline([
            ('convert_to_str', FunctionTransformer(column_as_string, validate=False)),
            ('tfidf', TfidfVectorizer())
        ]), 'body'),
        ('extracted_urls', Pipeline([
            ('convert_to_str', FunctionTransformer(column_as_string, validate=False)),
            ('tfidf', TfidfVectorizer())
        ]), 'extracted_urls'),
        ('subject', Pipeline([
            ('convert_to_str', FunctionTransformer(column_as_string, validate=False)),
            ('tfidf', TfidfVectorizer())
        ]), 'subject'),
        ('sender', Pipeline([
            ('convert_to_str', FunctionTransformer(column_as_string, validate=False)),
            ('tfidf', TfidfVectorizer())
        ]), 'sender')
    ]
)

# Create the Random Forest pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('nb', RandomForestClassifier())
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'nb__n_estimators': [100, 200, 500],
    'nb__max_depth': [10, 20, None],
    'nb__min_samples_split': [2, 5, 10],
    'nb__min_samples_leaf': [1, 2, 4],
    'nb__bootstrap': [True, False]
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Make predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Evaluate the model using classification report and ROC AUC score
print("Train Set Performance:")
print(classification_report(y_train, y_pred_train))
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred_test))

# Predict probabilities for ROC curve
pred_prob_train = best_model.predict_proba(X_train)[:, 1]
pred_prob_test = best_model.predict_proba(X_test)[:, 1]

# Calculate ROC AUC score
roc_auc_train = roc_auc_score(y_train, y_pred_train)
roc_auc_test = roc_auc_score(y_test, y_pred_test)
print("\nTrain ROC AUC:", roc_auc_train)
print("Test ROC AUC:", roc_auc_test)

# Plot the ROC curve
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, pred_prob_train)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_prob_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_train, tpr_train, label="Train ROC AUC: {:.2f}".format(roc_auc_train))
plt.plot(fpr_test, tpr_test, label="Test ROC AUC: {:.2f}".format(roc_auc_test))
plt.legend()
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# Confusion matrix
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

# Train confusion matrix
sns.heatmap(cm_train, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], cmap="Oranges", fmt='.4g', ax=ax[0])
ax[0].set_xlabel("Predicted Label")
ax[0].set_ylabel("True Label")
ax[0].set_title("Train Confusion Matrix")

# Test confusion matrix
sns.heatmap(cm_test, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], cmap="Oranges", fmt='.4g', ax=ax[1])
ax[1].set_xlabel("Predicted Label")
ax[1].set_ylabel("True Label")
ax[1].set_title("Test Confusion Matrix")

plt.tight_layout()
plt.show()

# Test with new email data
new_email_body = """hey mate, i saw your pull request, its good actually, but i feel some changes can be done, look on that and fix that"""
new_email_body = clean_text(new_email_body)
new_email_urls = extract_urls(new_email_body)

# Create a DataFrame similar to the one used for training
new_email_df = pd.DataFrame({
    'body': [new_email_body],
    'extracted_urls': [' '.join(new_email_urls)],
    'subject': ['new changes'],
    'sender': ['example@example.com']
})

# Make a prediction for the new email
prediction = best_model.predict(new_email_df)

# Output the prediction
if prediction == 0:
    print("This is not a spam Email!")
else:
    print("This is a Spam Email!")
