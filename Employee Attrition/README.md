# Employee Attrition Prediction using NLP & LangChain

## Overview
This project predicts employee attrition using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. It leverages **LangChain's OpenAI Embeddings** to transform employee reviews into numerical vectors and trains a **Random Forest classifier** to predict attrition.

## Features
✅ Uses **OpenAI’s NLP embeddings** for better accuracy.
✅ Employs a **Random Forest model** for classification.
✅ Supports **real-time predictions** based on employee reviews.
✅ **Easily extendable** to include retrieval-based analysis using **FAISS**.

---

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/employee-attrition.git
   cd employee-attrition
   ```
2. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn joblib openai langchain
   ```
3. **Set OpenAI API Key:** Replace `'your-api-key'` in the script with your actual API key.

---

## Dataset
The dataset should be a CSV file (`employee_reviews.csv`) with the following columns:
- `Review`: Employee's textual feedback.
- `Attrition`: Binary target variable (1 = Attrition, 0 = No Attrition).

---

## Code Explanation
### 1. **Preprocessing the Text Data**
```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text
```
This function:
- Converts text to **lowercase**.
- Removes **special characters** and **numbers**.

### 2. **Loading and Processing the Dataset**
```python
df = pd.read_csv("employee_reviews.csv")
df['Cleaned_Review'] = df['Review'].apply(preprocess_text)
```
- Loads the dataset from a CSV file.
- Cleans the `Review` column using `preprocess_text()`.

### 3. **Generating Text Embeddings using LangChain**
```python
embedding_model = OpenAIEmbeddings()
X = np.array([embedding_model.embed_query(text) for text in df['Cleaned_Review']])
y = df['Attrition']
```
- Converts text into **numerical embeddings** using **OpenAI embeddings**.

### 4. **Training the Machine Learning Model**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
- Splits data into **training (80%)** and **testing (20%)**.
- Trains a **Random Forest model**.

### 5. **Evaluating Model Performance**
```python
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))
```
- Computes **accuracy** and **classification metrics**.

### 6. **Saving the Trained Model**
```python
joblib.dump(model, "attrition_model.pkl")
```
- Saves the model for future use.

### 7. **Real-Time Prediction**
```python
def predict_attrition(review):
    review = preprocess_text(review)
    review_vectorized = np.array([embedding_model.embed_query(review)])
    prediction = model.predict(review_vectorized)
    return "Attrition Risk" if prediction[0] == 1 else "No Attrition Risk"
```
- **Cleans** the input review.
- **Converts it to an embedding**.
- **Predicts attrition risk** using the trained model.

### 8. **Example Usage**
```python
sample_review = "I feel stuck in my career with no growth opportunities."
print(predict_attrition(sample_review))
```
- Predicts **attrition risk** for a given review.

-

