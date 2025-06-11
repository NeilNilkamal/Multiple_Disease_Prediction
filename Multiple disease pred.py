import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Health Assistant",
                    layout="wide",
                    page_icon="🧑‍⚕️")

# --- Get the working directory ---
working_dir = os.path.dirname(os.path.abspath(__file__))

# --- Define history file path ---
HISTORY_FILE = os.path.join(working_dir, 'history_data.pkl')

# --- Function to load/save history data ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'rb') as f:
            return pickle.load(f)
    return {'Diabetes': [], 'Heart Disease': [], 'Parkinsons': []}

def save_history(data):
    with open(HISTORY_FILE, 'wb') as f:
        pickle.dump(data, f)

# Load existing history data
history_data = load_history()

# --- Loading the saved ML models ---
try:
    diabetes_model = pickle.load(open(f'{working_dir}/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open(f'{working_dir}/parkinsons_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Error: One or more machine learning model files (.sav) not found. "
             "Please ensure 'diabetes_model.sav', 'heart_disease_model.sav', and "
             "'parkinsons_model.sav' are in the same directory as this script.")
    st.stop() # Stop the app if models are missing

# --- Function to clear form inputs ---
def clear_form_inputs(keys):
    for key in keys:
        # Check if the key exists in session_state before trying to delete
        # This prevents errors if a key wasn't initialized (e.g., if user didn't interact with it)
        if key in st.session_state:
            del st.session_state[key]
    st.rerun() # Rerun to clear the inputs on the UI

# --- Sidebar for navigation ---
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                            ['Home',
                             'Diabetes Prediction',
                             'Heart Disease Prediction',
                             'Parkinsons Prediction',
                             'Prediction History',
                             'Health Resources'], # Added Health Resources
                            menu_icon='hospital-fill',
                            icons=['house', 'activity', 'heart', 'person', 'bar-chart-line', 'book'], # Added 'book' icon
                            default_index=0)

# --- Home Page ---
if selected == 'Home':
    st.title('Health Assistant')
    st.write("Welcome to your personal Health Assistant! This application helps you predict the likelihood of certain diseases based on your input parameters.")
    st.write("Navigate through the sidebar to access different prediction models or to view your past prediction history.")
    # st.image(os.path.join(working_dir, 'health_banner.jpg'), use_column_width=True, caption="Your Health, Our Priority") # Optional: Add an image if you have one. Replace 'health_banner.jpg' with your image file path.
    st.markdown("---")
    st.markdown("### How to Use:")
    st.markdown("- Select a disease prediction from the sidebar.")
    st.markdown("- Enter the required health parameters accurately.")
    st.markdown("- Click 'Test Result' to get your prediction.")
    st.markdown("- Your predictions are automatically saved to 'Prediction History' for future reference.")
    st.markdown("- You can download your history data and outcome summaries in the 'Prediction History' tab.")
    st.markdown("---")
    st.warning("Disclaimer: This application is for informational and educational purposes only. It is not intended to provide medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns.")


# --- Diabetes Prediction Page ---
elif selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using ML')

    # Define input keys for clearing (make sure these match the 'key' argument in st.number_input/selectbox)
    diabetes_input_keys = ['Pregnancies_D', 'Glucose_D', 'BloodPressure_D', 'SkinThickness_D',
                           'Insulin_D', 'BMI_D', 'DiabetesPedigreeFunction_D', 'Age_D']

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, format="%i", key="Pregnancies_D")
        st.info("Number of times pregnant.")
    with col2:
        Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0.0, max_value=300.0, value=120.0, format="%.1f", key="Glucose_D")
        st.info("Plasma glucose concentration a 2 hours after an oral glucose tolerance test (OGTT).")
    with col3:
        BloodPressure = st.number_input('Blood Pressure (mmHg)', min_value=0.0, max_value=200.0, value=70.0, format="%.1f", key="BloodPressure_D")
        st.info("Diastolic blood pressure.")
    with col1:
        SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0.0, max_value=100.0, value=20.0, format="%.1f", key="SkinThickness_D")
        st.info("Triceps skin fold thickness.")
    with col2:
        Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0.0, max_value=900.0, value=80.0, format="%.1f", key="Insulin_D")
        st.info("2-Hour serum insulin.")
    with col3:
        BMI = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f", key="BMI_D")
        st.info("Body Mass Index.")
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, format="%.3f", key="DiabetesPedigreeFunction_D")
        st.info("A function that scores likelihood of diabetes based on family history.")
    with col2:
        Age = st.number_input('Age of the Person', min_value=0, max_value=120, value=30, format="%i", key="Age_D")
        st.info("Age in years.")

    diab_diagnosis = ''

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button('Diabetes Test Result', help="Click to get prediction"):
            try:
                user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                              BMI, DiabetesPedigreeFunction, Age]

                diab_prediction = diabetes_model.predict([user_input])

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'
                
                st.success(diab_diagnosis)

                # --- Store prediction history ---
                history_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'inputs': {
                        'Pregnancies': Pregnancies,
                        'Glucose': Glucose,
                        'BloodPressure': BloodPressure,
                        'SkinThickness': SkinThickness,
                        'Insulin': Insulin,
                        'BMI': BMI,
                        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                        'Age': Age
                    },
                    'result': diab_diagnosis
                }
                history_data['Diabetes'].append(history_entry)
                save_history(history_data)

            except Exception as e:
                st.error(f"Please check your input values or ensure they are within a reasonable range. An error occurred: {e}")
    with col_btn2:
        if st.button('Clear Inputs', help="Click to clear all input fields for Diabetes Prediction", key="clear_diabetes_inputs"):
            clear_form_inputs(diabetes_input_keys)


# --- Heart Disease Prediction Page ---
elif selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using ML')

    # Define input keys for clearing
    heart_input_keys = ['age_hd_D', 'sex_hd_D', 'cp_hd_D', 'trestbps_hd_D', 'chol_hd_D',
                        'fbs_hd_D', 'restecg_hd_D', 'thalach_hd_D', 'exang_hd_D',
                        'oldpeak_hd_D', 'slope_hd_D', 'ca_hd_D', 'thal_hd_D']

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=50, format="%i", key="age_hd_D")
        st.info("Age of the patient in years.")
    with col2:
        sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="sex_hd_D")
        st.info("Gender (0 = Female, 1 = Male).")
    with col3:
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}[x], key="cp_hd_D")
        st.info("Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic).")
    with col1:
        trestbps = st.number_input('Resting Blood Pressure (mmHg)', min_value=0.0, max_value=250.0, value=120.0, format="%.1f", key="trestbps_hd_D")
        st.info("Resting blood pressure.")
    with col2:
        chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=0.0, max_value=600.0, value=200.0, format="%.1f", key="chol_hd_D")
        st.info("Serum cholestoral in mg/dl.")
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="fbs_hd_D")
        st.info("Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes).")
    with col1:
        restecg = st.selectbox('Resting Electrocardiographic results', options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x], key="restecg_hd_D")
        st.info("Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy).")
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0.0, max_value=250.0, value=150.0, format="%.1f", key="thalach_hd_D")
        st.info("Maximum heart rate achieved during exercise.")
    with col3:
        exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="exang_hd_D")
        st.info("Exercise induced angina (0 = No, 1 = Yes).")
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0, format="%.1f", key="oldpeak_hd_D")
        st.info("ST depression induced by exercise relative to rest.")
    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', options=[0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x], key="slope_hd_D")
        st.info("The slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping).")
    with col3:
        ca = st.selectbox('Major vessels colored by flourosopy', options=[0, 1, 2, 3], format_func=lambda x: f"{x} vessels", key="ca_hd_D")
        st.info("Number of major vessels (0-3) colored by flourosopy.")
    with col1:
        thal = st.selectbox('Thal', options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}[x], key="thal_hd_D")
        st.info("Thalassemia (0: normal, 1: fixed defect, 2: reversible defect).")

    heart_diagnosis = ''

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button('Heart Disease Test Result', help="Click to get prediction"):
            try:
                user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

                heart_prediction = heart_disease_model.predict([user_input])

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'
                
                st.success(heart_diagnosis)

                # --- Store prediction history ---
                history_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'inputs': {
                        'age': age, 'sex': sex, 'cp': cp, 
                        'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 
                        'restecg': restecg, 'thalach': thalach, 'exang': exang, 
                        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
                    },
                    'result': heart_diagnosis
                }
                history_data['Heart Disease'].append(history_entry)
                save_history(history_data)

            except Exception as e:
                st.error(f"Please check your input values or ensure they are within a reasonable range. An error occurred: {e}")
    with col_btn2:
        if st.button('Clear Inputs', key="clear_heart_inputs", help="Click to clear all input fields for Heart Disease Prediction"):
            clear_form_inputs(heart_input_keys)

# --- Parkinson's Prediction Page ---
elif selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction using ML")

    # Define input keys for clearing
    parkinsons_input_keys = ['fo_P', 'fhi_P', 'flo_P', 'Jitter_percent_P', 'Jitter_Abs_P',
                             'RAP_P', 'PPQ_P', 'DDP_P','Shimmer_P', 'Shimmer_dB_P', 'APQ3_P', 'APQ5_P',
                             'APQ_P', 'DDA_P', 'NHR_P', 'HNR_P', 'RPDE_P', 'DFA_P', 'spread1_P',
                             'spread2_P', 'D2_P', 'PPE_P']

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, value=120.0, format="%.3f", key="fo_P")
        st.info("Average vocal fundamental frequency.")
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, value=150.0, format="%.3f", key="fhi_P")
        st.info("Maximum vocal fundamental frequency.")
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, value=90.0, format="%.3f", key="flo_P")
        st.info("Minimum vocal fundamental frequency.")
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, value=0.005, format="%.5f", key="Jitter_percent_P")
        st.info("Measure of variation in fundamental frequency.")
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, value=0.00005, format="%.7f", key="Jitter_Abs_P")
        st.info("Absolute measure of variation in fundamental frequency.")
    with col1:
        RAP = st.number_input('MDVP:RAP', min_value=0.0, value=0.002, format="%.5f", key="RAP_P")
        st.info("Relative Amplitude Perturbation.")
    with col2:
        PPQ = st.number_input('MDVP:PPQ', min_value=0.0, value=0.003, format="%.5f", key="PPQ_P")
        st.info("Five-point Period Perturbation Quotient.")
    with col3:
        DDP = st.number_input('Jitter:DDP', min_value=0.0, value=0.006, format="%.5f", key="DDP_P")
        st.info("Average absolute difference of differences between adjacent periods.")
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, value=0.02, format="%.5f", key="Shimmer_P")
        st.info("Measure of amplitude variation.")
    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, value=0.2, format="%.3f", key="Shimmer_dB_P")
        st.info("Measure of amplitude variation in dB.")
    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, value=0.01, format="%.5f", key="APQ3_P")
        st.info("Three-point Amplitude Perturbation Quotient.")
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, value=0.015, format="%.5f", key="APQ5_P")
        st.info("Five-point Amplitude Perturbation Quotient.")
    with col3:
        APQ = st.number_input('MDVP:APQ', min_value=0.0, value=0.02, format="%.5f", key="APQ_P")
        st.info("Average Period Perturbation Quotient.")
    with col4:
        DDA = st.number_input('Shimmer:DDA', min_value=0.0, value=0.03, format="%.5f", key="DDA_P")
        st.info("Average absolute difference of differences between adjacent amplitudes.")
    with col5:
        NHR = st.number_input('NHR', min_value=0.0, value=0.01, format="%.5f", key="NHR_P")
        st.info("Noise-to-Harmonics Ratio.")
    with col1:
        HNR = st.number_input('HNR', min_value=0.0, value=20.0, format="%.3f", key="HNR_P")
        st.info("Harmonics-to-Noise Ratio.")
    with col2:
        RPDE = st.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.5, format="%.5f", key="RPDE_P")
        st.info("Recurrence Period Density Entropy.")
    with col3:
        DFA = st.number_input('DFA', min_value=0.0, max_value=1.0, value=0.7, format="%.5f", key="DFA_P")
        st.info("Detrended Fluctuation Analysis.")
    with col4:
        spread1 = st.number_input('spread1', min_value=-10.0, max_value=0.0, value=-5.0, format="%.5f", key="spread1_P")
        st.info("Nonlinear dynamical complexity measure 1.")
    with col5:
        spread2 = st.number_input('spread2', min_value=0.0, max_value=1.0, value=0.2, format="%.5f", key="spread2_P")
        st.info("Nonlinear dynamical complexity measure 2.")
    with col1:
        D2 = st.number_input('D2', min_value=0.0, max_value=5.0, value=2.0, format="%.5f", key="D2_P")
        st.info("Nonlinear dynamical complexity measure 3.")
    with col2:
        PPE = st.number_input('PPE', min_value=0.0, max_value=1.0, value=0.2, format="%.5f", key="PPE_P")
        st.info("Pitch Period Entropy.")

    parkinsons_diagnosis = ''
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Parkinson's Test Result", help="Click to get prediction"):
            try:
                user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                              RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                              APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

                parkinsons_prediction = parkinsons_model.predict([user_input])

                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "The person has Parkinson's disease"
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson's disease"
                
                st.success(parkinsons_diagnosis)

                # --- Store prediction history ---
                history_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'inputs': {
                        'fo': fo, 'fhi': fhi, 'flo': flo, 'Jitter_percent': Jitter_percent, 
                        'Jitter_Abs': Jitter_Abs, 'RAP': RAP, 'PPQ': PPQ, 'DDP': DDP, 
                        'Shimmer': Shimmer, 'Shimmer_dB': Shimmer_dB, 'APQ3': APQ3, 
                        'APQ5': APQ5, 'APQ': APQ, 'DDA': DDA, 'NHR': NHR, 
                        'HNR': HNR, 'RPDE': RPDE, 'DFA': DFA, 'spread1': spread1, 
                        'spread2': spread2, 'D2': D2, 'PPE': PPE
                    },
                    'result': parkinsons_diagnosis
                }
                history_data['Parkinsons'].append(history_entry)
                save_history(history_data)

            except Exception as e:
                st.error(f"Please check your input values or ensure they are within a reasonable range. An error occurred: {e}")
    with col_btn2:
        if st.button('Clear Inputs', key="clear_parkinsons_inputs", help="Click to clear all input fields for Parkinson's Prediction"):
            clear_form_inputs(parkinsons_input_keys)

### Prediction History Tab

elif selected == 'Prediction History':
    st.title('Your Health Prediction History')
    st.markdown("Review your past prediction results and observe trends in your health data.")

    # Dropdown to select disease type
    disease_to_show = st.selectbox(
        "Select Disease to View History:",
        ['Diabetes', 'Heart Disease', 'Parkinsons'],
        key="history_disease_select"
    )

    if disease_to_show in history_data and history_data[disease_to_show]:
        st.subheader(f"History for {disease_to_show}")
        
        # Convert history data to DataFrame for easier handling and display
        df_history = pd.DataFrame(history_data[disease_to_show])
        
        # Convert 'timestamp' to datetime objects for plotting
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        
        # Display raw data
        st.markdown("#### Raw Prediction Data")
        st.dataframe(df_history[['timestamp', 'result', 'inputs']]) # Display inputs as a dictionary
        
        # --- Download Raw History Data ---
        # Prepare data for download: flatten 'inputs' dictionary into separate columns
        df_history_flat = df_history.copy()
        df_history_flat = pd.concat([df_history_flat.drop('inputs', axis=1), df_history_flat['inputs'].apply(pd.Series)], axis=1)

        csv_data = df_history_flat.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {disease_to_show} History Data as CSV",
            data=csv_data,
            file_name=f"{disease_to_show}_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            help=f"Download all historical {disease_to_show} prediction data as a CSV file, including input parameters."
        )


        st.markdown("#### Visual Trends (Input Parameters)")

        # --- Plotting logic based on disease type (Input Parameters) ---
        df_plot = df_history.copy()
        df_plot = df_plot.set_index('timestamp') # Set timestamp as index for plotting

        if disease_to_show == 'Diabetes':
            try:
                # Safely extract numeric values, coercing errors to NaN and dropping
                df_plot['Glucose'] = df_plot['inputs'].apply(lambda x: x.get('Glucose')).apply(pd.to_numeric, errors='coerce')
                df_plot['BMI'] = df_plot['inputs'].apply(lambda x: x.get('BMI')).apply(pd.to_numeric, errors='coerce')
                df_plot['BloodPressure'] = df_plot['inputs'].apply(lambda x: x.get('BloodPressure')).apply(pd.to_numeric, errors='coerce')

                # Select only the columns for plotting and drop rows with any NaN values
                df_plot_numeric = df_plot[['Glucose', 'BMI', 'BloodPressure']].dropna()
                
                if not df_plot_numeric.empty:
                    st.line_chart(df_plot_numeric)
                    st.caption("Trends in Glucose, BMI, and Blood Pressure over time.")
                else:
                    st.info(f"Not enough valid numerical data for plotting for {disease_to_show} yet. Make more predictions.")
            except Exception as e:
                st.warning(f"Could not generate input trend chart for Diabetes. Error: {e}")


        elif disease_to_show == 'Heart Disease':
            try:
                df_plot['Cholesterol'] = df_plot['inputs'].apply(lambda x: x.get('chol')).apply(pd.to_numeric, errors='coerce')
                df_plot['Thalach'] = df_plot['inputs'].apply(lambda x: x.get('thalach')).apply(pd.to_numeric, errors='coerce')
                df_plot['RestingBP'] = df_plot['inputs'].apply(lambda x: x.get('trestbps')).apply(pd.to_numeric, errors='coerce')

                df_plot_numeric = df_plot[['Cholesterol', 'Thalach', 'RestingBP']].dropna()
                
                if not df_plot_numeric.empty:
                    st.line_chart(df_plot_numeric)
                    st.caption("Trends in Cholesterol, Max Heart Rate, and Resting Blood Pressure over time.")
                else:
                    st.info(f"Not enough valid numerical data for plotting for {disease_to_show} yet. Make more predictions.")
            except Exception as e:
                st.warning(f"Could not generate input trend chart for Heart Disease. Error: {e}")

        elif disease_to_show == 'Parkinsons':
            try:
                df_plot['HNR'] = df_plot['inputs'].apply(lambda x: x.get('HNR')).apply(pd.to_numeric, errors='coerce')
                df_plot['Jitter_percent'] = df_plot['inputs'].apply(lambda x: x.get('Jitter_percent')).apply(pd.to_numeric, errors='coerce')
                df_plot['Shimmer'] = df_plot['inputs'].apply(lambda x: x.get('Shimmer')).apply(pd.to_numeric, errors='coerce')

                df_plot_numeric = df_plot[['HNR', 'Jitter_percent', 'Shimmer']].dropna()
                
                if not df_plot_numeric.empty:
                    st.line_chart(df_plot_numeric)
                    st.caption("Trends in HNR, Jitter, and Shimmer over time (voice parameters).")
                else:
                    st.info(f"Not enough valid numerical data for plotting for {disease_to_show} yet. Make more predictions.")
            except Exception as e:
                st.warning(f"Could not generate input trend chart for Parkinsons. Error: {e}")

        # --- Prediction Outcome Status Report ---
        st.markdown("#### Prediction Outcome Summary")
        if not df_history.empty:
            outcome_counts = df_history['result'].value_counts()
            st.bar_chart(outcome_counts)
            st.caption(f"Count of each prediction outcome for {disease_to_show}.")

            # --- Download Outcome Summary Data ---
            outcome_csv = outcome_counts.reset_index(name='Count').to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download {disease_to_show} Outcome Summary as CSV",
                data=outcome_csv,
                file_name=f"{disease_to_show}_outcome_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                help=f"Download the summarized prediction outcomes for {disease_to_show}."
            )
        else:
            st.info(f"No prediction outcomes to summarize for {disease_to_show} yet.")

    else:
        st.info(f"No prediction history available for {disease_to_show} yet. Make some predictions first!")

# --- Health Resources and Tips Page ---
elif selected == 'Health Resources':
    st.title('General Health Resources and Tips')
    st.write("Here you'll find general information and tips to help maintain a healthy lifestyle and understand more about the diseases this app focuses on. Remember, this information is not a substitute for professional medical advice.")
    
    st.markdown("---")

    st.subheader("General Health & Wellness")
    st.markdown("""
    - **Balanced Diet:** Focus on whole foods, fruits, vegetables, lean proteins, and healthy fats. Limit processed foods, sugary drinks, and excessive saturated/trans fats.
    - **Regular Exercise:** Aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week, along with muscle-strengthening activities twice a week.
    - **Adequate Sleep:** Prioritize 7-9 hours of quality sleep per night. Good sleep is crucial for physical and mental well-being.
    - **Stress Management:** Practice stress-reducing techniques like meditation, yoga, deep breathing, or hobbies you enjoy.
    - **Stay Hydrated:** Drink plenty of water throughout the day.
    - **Avoid Smoking and Limit Alcohol:** These habits significantly increase the risk of numerous health problems.
    - **Regular Check-ups:** Visit your doctor for routine physicals and screenings, even if you feel healthy. Early detection is key.
    """)

    st.subheader("Tips for Diabetes Management & Prevention")
    st.markdown("""
    - **Monitor Blood Sugar:** If you have diabetes, regularly check your blood glucose levels as advised by your doctor.
    - **Carbohydrate Counting:** Learn about carbohydrates and how they affect blood sugar.
    - **Portion Control:** Manage your food portions to maintain a healthy weight.
    - **Fiber Intake:** Increase dietary fiber to help control blood sugar and improve digestion.
    - **Regular Physical Activity:** Exercise helps improve insulin sensitivity.
    - **Medication Adherence:** Take prescribed medications as directed by your healthcare provider.
    """)

    st.subheader("Tips for Heart Health")
    st.markdown("""
    - **Maintain Healthy Blood Pressure:** Regularly monitor your blood pressure and follow your doctor's recommendations for management (diet, exercise, medication).
    - **Manage Cholesterol Levels:** Reduce intake of saturated and trans fats. Focus on foods that can lower cholesterol like oats, nuts, and fatty fish.
    - **Healthy Weight:** Obesity is a major risk factor for heart disease.
    - **Limit Sodium Intake:** High sodium can contribute to high blood pressure.
    - **Omega-3 Fatty Acids:** Include sources like salmon, flaxseeds, and walnuts in your diet.
    - **Quit Smoking:** Smoking is a leading cause of heart disease.
    """)

    st.subheader("Tips for Parkinson's Disease (General Information)")
    st.markdown("""
    - **Exercise:** Regular physical activity, including aerobic exercise, strength training, balance, and flexibility exercises, can help manage symptoms and improve mobility.
    - **Balanced Diet:** While no specific diet cures Parkinson's, a balanced diet rich in fruits, vegetables, whole grains, and lean proteins can support overall health.
    - **Speech Therapy:** Can help with voice changes and swallowing difficulties.
    - **Occupational Therapy:** Can help adapt daily tasks and maintain independence.
    - **Physical Therapy:** Focuses on improving balance, gait, and flexibility.
    - **Medication Management:** Adhering to prescribed medications is crucial for symptom management.
    """)
    st.warning("Please remember that the information provided here is for general knowledge and informational purposes only, and does not constitute medical advice. Always consult with a qualified healthcare professional for personalized advice, diagnosis, or treatment regarding any medical condition.")
