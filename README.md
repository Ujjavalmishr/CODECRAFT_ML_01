# CODECRAFT_ML_01

# 🏡 House Price Predictor

A simple machine learning project that predicts house prices based on square footage, number of bedrooms, and number of bathrooms.  
The project uses a **Linear Regression** model built with **Scikit-learn** and provides both a **Streamlit UI** and a **Flask API** for interaction.

---

## 📂 Project Structure

├── app.py # Streamlit web app for predictions
├── train_and_save_model.py # Script to train and save model and imputer
├── model.pkl # Trained Linear Regression model
├── imputer.pkl # Imputer for handling missing values
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## 🚀 Features

- ✅ Predicts house prices using:
  - Square footage (`GrLivArea`)
  - Number of bedrooms (`BedroomAbvGr`)
  - Number of bathrooms (`FullBath`, `HalfBath`)
  - Year built, total rooms, garage area
- ✅ Preprocessing includes missing value handling
- ✅ Trained using **Scikit-learn's Linear Regression**
- ✅ Web app with **Streamlit UI**
- ✅ Optional: REST API interface with **Flask**

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Ujjavalmishr/CODECRAFT_ML_01.git
cd CodeCraft_ML_01

2. Install dependencies
All dependencies are listed in requirements.txt:
text
Copy
Edit
Flask==3.0.2
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
joblib==1.4.2

pip install -r requirements.txt


3. Train the model (if not already trained)
bash
Copy code
python train_and_save_model.py
This generates:

model.pkl — trained regression model

imputer.pkl — preprocessing object to handle missing values

4. Run the Streamlit app
bash
Copy code
streamlit run app.py
Visit: http://localhost:8501


🧠 Model Details
Algorithm: Linear Regression

Input Features:

Square footage (float)

Bedrooms (int)

Bathrooms (int)

Output: Predicted house price (float)

👨‍💻 Author
Ujjaval Mishra
BTECH CSE(AI) 
ABES Institue Of Technology, Ghaziabad
📧 ujjavalmishra439@gmail.com
🌐 https://github.com/Ujjavalmishr
