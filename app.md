import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model, scaler, and feature names
model = joblib.load('income_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("💼 Income Predictor")
st.markdown("Predict if income is >50K based on demographics and job details.")

# Input fields
age = st.slider("Age", 18, 68, 30)
education_num = st.slider("Education Level (numeric)", 1, 16, 10)
hours_per_week = st.slider("Hours worked per week", 1, 99, 40)
capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.number_input("Capital Loss", 0, 99999, 0)

# Categorical options
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay'])
marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                                         'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                         'Transport-moving', 'Priv-house-serv', 'Protective-serv'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
sex = st.selectbox("Sex", ['Male', 'Female'])
native_country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Other'])

# Derived features
age_group = 'Young' if age <= 30 else 'Mid' if age <= 45 else 'Senior'
is_educated = int(education_num > 12)

# Create a single row DataFrame from user input
input_data = {
    'age': age,
    'educational-num': education_num,
    'hours-per-week': hours_per_week,
    'capital-gain': capital_gain,
    'is_educated': is_educated,
    'age_group_' + age_group: 1,
    'workclass_' + workclass: 1,
    'marital-status_' + marital_status: 1,
    'occupation_' + occupation: 1,
    'relationship_' + relationship: 1,
    'race_' + race: 1,
    'sex_' + sex: 1,
    'native-country_' + native_country: 1,
}

# Set all other feature values to 0
full_input = pd.DataFrame([{
    col: input_data.get(col, 0) for col in feature_names
}])

# Scale input
scaled_input = scaler.transform(full_input)

# Predict
if st.button("🔮 Predict"):
    pred = model.predict(scaled_input)[0]
    result = ">50K" if pred == 1 else "<=50K"
    st.success(f"💰 Predicted Income: **{result}**")

# Education mapping table
st.markdown("---")
st.markdown("### 🎓 Education Level Mapping (for reference)")
education_table = """
<table>
<thead>
<tr><th>Education Num</th><th>Education Level</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>Preschool</td></tr>
<tr><td>2</td><td>1st-4th</td></tr>
<tr><td>3</td><td>5th-6th</td></tr>
<tr><td>4</td><td>7th-8th</td></tr>
<tr><td>5</td><td>9th</td></tr>
<tr><td>6</td><td>10th</td></tr>
<tr><td>7</td><td>11th</td></tr>
<tr><td>8</td><td>12th</td></tr>
<tr><td>9</td><td>HS-grad</td></tr>
<tr><td>10</td><td>Some-college</td></tr>
<tr><td>11</td><td>Assoc-voc</td></tr>
<tr><td>12</td><td>Assoc-acdm</td></tr>
<tr><td>13</td><td>Bachelors</td></tr>
<tr><td>14</td><td>Masters</td></tr>
<tr><td>15</td><td>Doctorate</td></tr>
<tr><td>16</td><td>Prof-school</td></tr>
</tbody>
</table>
"""
st.markdown(education_table, unsafe_allow_html=True)
