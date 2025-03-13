# ML Model Finder

Welcome to the **ML Model Finder** project! This application allows you to upload your dataset (in CSV or Excel format) and automatically discover the best machine learning model for your data. Below is a walkthrough of the application's interface and functionality.

---

## Screenshots and Descriptions

### 1. **Upload Your Dataset**
![Upload Your Dataset](images/Screenshot%202025-03-13%20at%2023.46.50.png)

- **Description**: The landing page of the application where users can upload their dataset. Supported formats include CSV and Excel files, with a maximum file size of 100MB.
- **Features**:
  - Drag and drop functionality for easy file upload.
  - A list of recent uploads for quick access to previously analyzed datasets.

---

### 2. **Dataset Uploaded Successfully**
![Dataset Uploaded Successfully](images/Screenshot%202025-03-13%20at%2023.47.23.png)

- **Description**: After uploading a dataset (e.g., `diabetes.csv`, `hearts.csv`), the application displays the filename and a list of recent uploads.
- **Features**:
  - Confirmation of successful upload.
  - Navigation buttons to proceed to the analysis or go back to upload a new dataset.

---

### 3. **Analysis Results - Summary**
![Analysis Results - Summary](images/Screenshot%202025-03-13%20at%2023.47.49.png)

- **Description**: The **Summary** tab provides an overview of the dataset and the best-performing model.
- **Key Information**:
  - **Filename**: Name of the uploaded dataset.
  - **Rows and Columns**: Size of the dataset.
  - **Target Column**: The column used as the target variable for model training.
  - **Problem Type**: Classification or Regression.
  - **Best Model**: The model with the highest performance score (e.g., Logistic Regression with 76.83% accuracy).
  - **Target Distribution**: A visual representation of the target variable's distribution.

---

### 4. **Analysis Results - Model Comparison**
![Analysis Results - Model Comparison](images/Screenshot%202025-03-13%20at%2023.48.30.png)

- **Description**: The **Model Comparison** tab shows a comparison of different machine learning models based on their performance (accuracy).
- **Key Information**:
  - A bar chart comparing the accuracy of models like KNN, Decision Tree, SVM, AdaBoost, Random Forest, and Logistic Regression.
  - Logistic Regression is highlighted as the best-performing model with an accuracy of 76.83%.

---

### 5. **Analysis Results - Detailed Model Scores**
![Analysis Results - Detailed Model Scores](images/Screenshot%202025-03-13%20at%2023.48.38.png)

- **Description**: The **Detailed Model Scores** section provides a breakdown of model performance across 5-fold cross-validation.
- **Key Information**:
  - Performance scores for each fold and the average score for models like AdaBoost, Decision Tree, KNN, Logistic Regression, Random Forest, and SVM.
  - Logistic Regression has the highest average accuracy of 76.83%.

---

### 6. **Feature Importance - How much each feature affect the outcome (for Classification)**

- **Description**: The **Feature Importance** section provides a importance of each feature that affects on the outcome
- **Key Information**:
  - A table providing percentage about each feature importancr on the outcome


---

## How to Use

1. **Upload Your Dataset**: Drag and drop your dataset (CSV or Excel) into the upload area or click to browse your computer.
2. **Analyze**: The application will automatically analyze your dataset and train multiple machine learning models.
3. **View Results**: Explore the results in the **Summary**, **Data Insights**, **Model Comparison**, and **Feature Importance** tabs.
4. **Download or Save**: Save the best model or download the analysis report for further use.

---

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Regressor Models**: Linear Regression, SVR, RandomForestRegressor, AdaBoostRegressor, KNeighborsRegressor, DecisionTreeRegressor
- **Classification Models**: Logistic Regression, SVC, RandomForestClassifier, AdaBoostClassifier, KNeighborsClassifier, DecisionTreeClassifier
- **Deployment**: Render

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/SPGOHIL09/ML_Model_Comparsion.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the application:
   ```bash
   python run.py