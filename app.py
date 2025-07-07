# import streamlit as st
# import pandas as pd
# import pickle

# # Load model and imputer
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("imputer.pkl", "rb") as f:
#     imputer = pickle.load(f)

# st.title("🏡 House Price Predictor")

# uploaded_file = st.file_uploader("Upload a CSV with columns: GrLivArea, BedroomAbvGr, FullBath", type="csv")

# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)

#     # Validate required columns
#     required_cols = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
#     if all(col in data.columns for col in required_cols):
#         X = data[required_cols]
#         X_imputed = imputer.transform(X)
#         predictions = model.predict(X_imputed)
#         data['PredictedPrice'] = predictions

#         st.write("### 📊 Predictions:")
#         st.dataframe(data)

#         # Download button
#         csv = data.to_csv(index=False).encode('utf-8')
#         st.download_button("📥 Download CSV with Predictions", csv, "predictions.csv", "text/csv")
#     else:
#         st.error(f"❌ Your CSV must contain these columns: {required_cols}")


import streamlit as st
import pandas as pd
import pickle

# Load model and imputer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

# Streamlit page setup
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")

st.title("🏡 House Price Predictor")
st.markdown("Upload a CSV file with the following columns:")

required_cols = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'GarageArea', 'YearBuilt']
st.code(", ".join(required_cols))

# Sidebar info
with st.sidebar:
    st.header("📘 About the Model")
    st.markdown("""
    **Model:** Linear Regression  
    **Features used for prediction:**  
    - GrLivArea  
    - BedroomAbvGr  
    - FullBath  
    - HalfBath  
    - TotRmsAbvGrd  
    - GarageArea  
    - YearBuilt
    """)
    with st.expander("❓ How to use"):
        st.markdown("""
        1. Prepare your CSV with the required columns.  
        2. Upload it to see predicted prices.  
        3. Download the results as a CSV.
        """)

uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if all(col in data.columns for col in required_cols):
            X = data[required_cols]
            X_imputed = imputer.transform(X)
            predictions = model.predict(X_imputed)

            data['PredictedPrice'] = predictions

            st.success("✅ Predictions generated successfully!")
            st.dataframe(data)

            st.markdown(f"### 🧮 Average Predicted Price: **${round(data['PredictedPrice'].mean(), 2):,}**")

            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "house_price_predictions.csv", "text/csv")
        else:
            st.error(f"Missing required columns: {', '.join(required_cols)}")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin prediction.")
