# Employee Attrition & Department Prediction  

This Jupyter Notebook implements a **branched neural network** to help HR predict:  
1. **Employee Attrition** – Whether an employee is likely to leave the company.  
2. **Best-Fit Department** – The department where an employee is most suited.  

## **Dataset**  
The notebook uses an employee attrition dataset, loaded from an external CSV file.

## **Dependencies**  
Ensure you have the following libraries installed before running the notebook:  

```bash
pip install pandas numpy scikit-learn tensorflow
```

## **Workflow Overview**  

### **1. Data Preprocessing**
- Loads data using `pandas`
- Encodes categorical variables  
- Normalizes numerical features with `StandardScaler`
- Splits data into training and testing sets

### **2. Neural Network Model**
A **branched neural network** is implemented using **TensorFlow/Keras**:
- **Shared input layers** process general employee features.
- Two output branches:
  - **Attrition Prediction** – Uses a binary classification layer (Sigmoid activation).  
  - **Department Prediction** – Uses a multi-class classification layer (Softmax activation).  

### **3. Model Training & Evaluation**
- Compiles the model with appropriate loss functions (`binary_crossentropy` and `categorical_crossentropy`).
- Trains the model using training data.
- Evaluates model performance on test data.

## **Usage**
1. Open the notebook and run all cells sequentially.
2. The trained model can be used to make predictions on new employee data.

## **Possible Improvements**
- Hyperparameter tuning for better accuracy.
- Feature engineering to improve model performance.
- Implementing dropout layers to prevent overfitting.

