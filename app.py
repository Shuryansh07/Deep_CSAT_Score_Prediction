import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==========================================
# 1. PAGE CONFIGURATION & UI STYLING
# ==========================================
st.set_page_config(
    page_title="CSAT Predictor AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a better UI
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #1E3A8A; text-align: center; font-family: 'Helvetica Neue', sans-serif;}
    .stButton>button {width: 100%; border-radius: 8px; background-color: #1E3A8A; color: white; font-weight: bold;}
    .stButton>button:hover {background-color: #3B82F6; color: white;}
    .score-text {font-size: 24px; font-weight: bold; color: #10B981;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD ASSETS (CACHED FOR SPEED)
# ==========================================
@st.cache_resource
def load_assets():
    model = load_model('csat_ann_model.h5')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

# ==========================================
# 3. APP HEADER
# ==========================================
st.markdown("<h1>🛍️ E-Commerce Customer Satisfaction (CSAT) AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px; color: #6B7280;'>Predict customer satisfaction scores in real-time using Deep Learning.</p>", unsafe_allow_html=True)
st.write("---")

# ==========================================
# 4. USER INPUT SECTION (WITH COLUMNS)
# ==========================================
st.subheader("📝 Interaction Details")

# Creating a neat layout using 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📞 Channel & Category**")
    channel = st.selectbox("Channel Name", ['Outcall', 'Inbound', 'Email', 'Chat'])
    category = st.selectbox("Category", ['Product Queries', 'Order Related', 'Refund', 'Feedback', 'Other'])
    sub_category = st.selectbox("Sub-category", ['Life Insurance', 'Product Specific Information', 'Delay', 'Status', 'Other'])

with col2:
    st.markdown("**🧑‍💻 Agent Details**")
    tenure = st.selectbox("Agent Tenure Bucket", ['On Job Training', '0-30', '31-60', '61-90', '>90'])
    shift = st.selectbox("Agent Shift", ['Morning', 'Evening', 'Night', 'Split', 'Afternoon'])

with col3:
    st.markdown("**⏱️ Time Metrics**")
    reported_hour = st.slider("Hour Issue Reported (0-23)", min_value=0, max_value=23, value=14, help="Military time hour (e.g., 14 = 2 PM)")
    response_time = st.number_input("Response Time (Minutes)", min_value=0.0, max_value=10000.0, value=15.0, help="Time taken to respond to the customer")

st.write("---")

# ==========================================
# 5. PREDICTION LOGIC & RESULTS UI
# ==========================================
if st.button("🚀 Predict CSAT Score"):
    with st.spinner('Analyzing interaction patterns...'):
        
        # 1. Create a DataFrame from the inputs
        input_data = pd.DataFrame([[
            channel, category, sub_category, tenure, shift, response_time, reported_hour
        ]], columns=[
            'channel_name', 'category', 'Sub-category', 'Tenure Bucket', 'Agent Shift', 'Response_Time_Minutes', 'Reported_Hour'
        ])
        
        # 2. One-Hot Encode the inputs
        input_encoded = pd.get_dummies(input_data)
        
        # 3. Align with Training Columns (CRITICAL STEP)
        # This ensures the input has the exact same 70+ columns as the training data. Missing categories become 0.
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # 4. Scale the Data
        input_scaled = scaler.transform(input_aligned)
        
        # 5. Make the Prediction (Classification)
        pred_probs = model.predict(input_scaled)
        predicted_class = np.argmax(pred_probs, axis=1)[0]
        
        # Add 1 because our classes were 0-4 during training, but real CSAT is 1-5
        final_csat = predicted_class + 1 
        confidence = np.max(pred_probs) * 100

        # ==========================================
        # 6. BEAUTIFUL OUTPUT DISPLAY
        # ==========================================
        st.write("")
        st.subheader("📊 Prediction Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Predicted CSAT Score", value=f"{final_csat} / 5")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
        with res_col2:
            # Display Stars based on the score
            stars = "⭐" * final_csat
            st.markdown(f"<p style='font-size: 30px;'>{stars}</p>", unsafe_allow_html=True)
            
            # Contextual message
            if final_csat >= 4:
                st.success("✅ The customer is likely to be highly satisfied!")
            elif final_csat == 3:
                st.warning("⚠️ The customer might have a neutral experience. Room for improvement.")
            else:
                st.error("🚨 High risk of customer dissatisfaction! Immediate follow-up recommended.")