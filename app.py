import streamlit as st.

# Load all models
log_model = pickle.load(open('logistic_model.pkl', 'rb'))
dt_model = pickle.load(open('decision_tree_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))

st.title('Telco Customer Churn Predictor')
st.write('Enter customer details below to predict churn risk')

# Input fields
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])

with col2:
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges', 18.0, 120.0, 65.0)
    total_charges = st.number_input('Total Charges', 0.0, 10000.0, 1000.0)

if st.button('Predict Churn'):

    # Encode inputs same way as notebook
    gender_enc = 1 if gender == 'Male' else 0
    senior_enc = 1 if senior == 'Yes' else 0
    partner_enc = 1 if partner == 'Yes' else 0
    dependents_enc = 1 if dependents == 'Yes' else 0
    phone_enc = 1 if phone_service == 'Yes' else 0
    online_sec_enc = 2 if online_security == 'Yes' else (0 if online_security == 'No' else 1)
    online_bak_enc = 2 if online_backup == 'Yes' else (0 if online_backup == 'No' else 1)
    device_enc = 2 if device_protection == 'Yes' else (0 if device_protection == 'No' else 1)
    tech_enc = 2 if tech_support == 'Yes' else (0 if tech_support == 'No' else 1)
    tv_enc = 2 if streaming_tv == 'Yes' else (0 if streaming_tv == 'No' else 1)
    movies_enc = 2 if streaming_movies == 'Yes' else (0 if streaming_movies == 'No' else 1)
    paperless_enc = 1 if paperless_billing == 'Yes' else 0

    multiple_lines_no_service = 1 if multiple_lines == 'No phone service' else 0
    multiple_lines_yes = 1 if multiple_lines == 'Yes' else 0
    internet_fiber = 1 if internet_service == 'Fiber optic' else 0
    internet_no = 1 if internet_service == 'No' else 0
    contract_one = 1 if contract == 'One year' else 0
    contract_two = 1 if contract == 'Two year' else 0
    payment_credit = 1 if payment_method == 'Credit card (automatic)' else 0
    payment_electronic = 1 if payment_method == 'Electronic check' else 0
    payment_mailed = 1 if payment_method == 'Mailed check' else 0

    input_data = np.array([[gender_enc, senior_enc, partner_enc, dependents_enc,
                            tenure, phone_enc, online_sec_enc, online_bak_enc,
                            device_enc, tech_enc, tv_enc, movies_enc, paperless_enc,
                            monthly_charges, total_charges,
                            multiple_lines_no_service, multiple_lines_yes,
                            internet_fiber, internet_no,
                            contract_one, contract_two,
                            payment_credit, payment_electronic, payment_mailed]])

    # Predictions
    st.subheader('Predictions from all models')
    col3, col4 = st.columns(2)

    models = {
        'Logistic Regression': log_model,
        'Decision Tree': dt_model,
        'SVM': svm_model,
        'Random Forest': rf_model
    }

    for name, model in models.items():
        pred = model.predict(input_data)[0]
        if pred == 1:
            st.error(f'{name} → Will Churn ⚠️')
        else:
            st.success(f'{name} → Will Stay ✅')