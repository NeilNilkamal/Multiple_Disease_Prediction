import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
import pandas as pd
import sqlite3 # Import for SQLite database
import uuid # For generating unique IDs
import uuid
from datetime import datetime, timedelta # Import datetime and timedelta
from streamlit_cookies_controller import CookieController # For managing browser cookies

# --- Page Configuration ---
st.set_page_config(page_title="Health Assistant",
                    layout="wide",
                    page_icon="🧑‍⚕️")

# --- Get the working directory ---
working_dir = os.path.dirname(os.path.abspath(__file__))

# --- SQLite Database Setup ---
DB_FILE = os.path.join(working_dir, 'health_history.db')

def get_db_connection():
    """Establishes a connection to the SQLite database and creates the table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS user_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_uuid TEXT NOT NULL,
            disease_type TEXT NOT NULL,
            inputs TEXT, -- Store inputs as JSON string
            prediction_result TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    return conn

# --- Initialize or Retrieve User UUID ---
def get_user_session_uuid():
    """
    Retrieves a unique UUID from browser cookies. If not found, generates a new one
    and sets it as a cookie. Stores it in session_state for easy access.
    """
    controller = CookieController()
    session_uuid = controller.get("user_health_app_uuid") # Use a specific cookie name

    if not session_uuid:
        session_uuid = str(uuid.uuid4()) # Generate a new unique ID
        
        # Calculate expiry date for 90 days from now
        expires_date = datetime.now() + timedelta(days=90)
        
        # Set the cookie using the 'expires' argument with a datetime object
        controller.set("user_health_app_uuid", session_uuid, expires=expires_date)
        
        # st.session_state.user_uuid = session_uuid # Store in session_state for current run
    # Else, if session_uuid was retrieved from cookie, it's already set.
    
    # Always store in session_state so it's readily available throughout the current session
    st.session_state.user_uuid = session_uuid 
    return session_uuid

# Ensure UUID is generated/retrieved at the start of every session
# Call this once at the top level to ensure the UUID is available
current_user_uuid = get_user_session_uuid()
# st.sidebar.write(f"Your session ID: {current_user_uuid[:8]}...") # Optional: for debugging

# --- Modified History Functions to interact with SQLite ---
def load_history(user_uuid):
    """Loads prediction history for a specific user UUID from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT disease_type, inputs, prediction_result, timestamp FROM user_predictions WHERE session_uuid = ? ORDER BY timestamp DESC",
        (user_uuid,)
    )
    history_records = cursor.fetchall()
    conn.close()

    history_list = []
    for record in history_records:
        history_list.append({
            "Disease": record[0],
            "Inputs": eval(record[1]) if record[1] else {}, # Convert string back to dict, handle empty
            "Result": record[2],
            "Timestamp": pd.to_datetime(record[3])
        })
    return history_list

def save_history(user_uuid, disease_type, inputs, prediction_result):
    """Saves a new prediction record for a specific user UUID to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO user_predictions (session_uuid, disease_type, inputs, prediction_result, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_uuid, disease_type, str(inputs), prediction_result, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

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
    st.markdown("---")
    st.markdown("### How to Use:")
    st.markdown("- Select a disease prediction from the sidebar.")
    st.markdown("- Enter the required health parameters accurately.")
    st.markdown("- Click 'Test Result' to get your prediction.")
    st.markdown("- Your predictions are automatically saved to 'Prediction History' for future reference on *this device/browser*.")
    st.markdown("- You can download your history data and outcome summaries in the 'Prediction History' tab.")
    st.markdown("---")
    st.warning("Disclaimer: This application is for informational and educational purposes only. It is not intended to provide medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns.")


# --- Diabetes Prediction Page ---
elif selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using ML')

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
                user_input_dict = { # Store inputs as a dictionary for easier saving/retrieval
                    'Pregnancies': Pregnancies,
                    'Glucose': Glucose,
                    'BloodPressure': BloodPressure,
                    'SkinThickness': SkinThickness,
                    'Insulin': Insulin,
                    'BMI': BMI,
                    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                    'Age': Age
                }
                user_input_list = list(user_input_dict.values()) # Convert to list for model

                diab_prediction = diabetes_model.predict([user_input_list])

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'
                
                st.success(diab_diagnosis)

                # --- Store prediction history in DB ---
                save_history(st.session_state.user_uuid, 'Diabetes', user_input_dict, diab_diagnosis)

            except Exception as e:
                st.error(f"Please check your input values or ensure they are within a reasonable range. An error occurred: {e}")
    with col_btn2:
        if st.button('Clear Inputs', help="Click to clear all input fields for Diabetes Prediction", key="clear_diabetes_inputs"):
            clear_form_inputs(diabetes_input_keys)


# --- Heart Disease Prediction Page ---
elif selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using ML')

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
                user_input_dict = {
                    'age': age, 'sex': sex, 'cp': cp, 
                    'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 
                    'restecg': restecg, 'thalach': thalach, 'exang': exang, 
                    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
                }
                user_input_list = list(user_input_dict.values())

                heart_prediction = heart_disease_model.predict([user_input_list])

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'
                
                st.success(heart_diagnosis)

                # --- Store prediction history in DB ---
                save_history(st.session_state.user_uuid, 'Heart Disease', user_input_dict, heart_diagnosis)

            except Exception as e:
                st.error(f"Please check your input values or ensure they are within a reasonable range. An error occurred: {e}")
    with col_btn2:
        if st.button('Clear Inputs', key="clear_heart_inputs", help="Click to clear all input fields for Heart Disease Prediction"):
            clear_form_inputs(heart_input_keys)

# --- Parkinson's Prediction Page ---
elif selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction using ML")

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
                user_input_dict = {
                    'fo': fo, 'fhi': fhi, 'flo': flo, 'Jitter_percent': Jitter_percent, 
                    'Jitter_Abs': Jitter_Abs, 'RAP': RAP, 'PPQ': PPQ, 'DDP': DDP, 
                    'Shimmer': Shimmer, 'Shimmer_dB': Shimmer_dB, 'APQ3': APQ3, 
                    'APQ5': APQ5, 'APQ': APQ, 'DDA': DDA, 'NHR': NHR, 
                    'HNR': HNR, 'RPDE': RPDE, 'DFA': DFA, 'spread1': spread1, 
                    'spread2': spread2, 'D2': D2, 'PPE': PPE
                }
                user_input_list = list(user_input_dict.values())

                parkinsons_prediction = parkinsons_model.predict([user_input_list])

                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "The person has Parkinson's disease"
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson's disease"
                
                st.success(parkinsons_diagnosis)

                # --- Store prediction history in DB ---
                save_history(st.session_state.user_uuid, 'Parkinsons', user_input_dict, parkinsons_diagnosis)

            except Exception as e:
                st.error(f"Please check your input values or ensure they are within a reasonable range. An error occurred: {e}")
    with col_btn2:
        if st.button('Clear Inputs', key="clear_parkinsons_inputs", help="Click to clear all input fields for Parkinson's Prediction"):
            clear_form_inputs(parkinsons_input_keys)

### Prediction History Tab

elif selected == 'Prediction History':
    st.title('Your Health Prediction History')
    st.markdown("Review your past prediction results and observe trends in your health data.")
    st.write("This history is unique to this specific browser and device.")

    # Dropdown to select disease type
    # For history display, we need to load all history first, then filter for display
    all_user_history = load_history(st.session_state.user_uuid)
    
    if not all_user_history:
        st.info("No prediction history available for this device yet. Make some predictions first!")
    else:
        # Get unique disease types from the loaded history for the dropdown
        unique_diseases = sorted(list(set([entry['Disease'] for entry in all_user_history])))
        if not unique_diseases: # Fallback if no diseases yet (shouldn't happen with existing history)
            unique_diseases = ['Diabetes', 'Heart Disease', 'Parkinsons']


        disease_to_show = st.selectbox(
            "Select Disease to View History:",
            unique_diseases,
            key="history_disease_select"
        )

        # Filter the loaded history based on user selection
        filtered_history = [entry for entry in all_user_history if entry['Disease'] == disease_to_show]

        if filtered_history:
            st.subheader(f"History for {disease_to_show}")
            
            # Convert filtered history data to DataFrame for easier handling and display
            df_history = pd.DataFrame(filtered_history)
            
            # Display raw data
            st.markdown("#### Raw Prediction Data")
            # Display inputs as a dictionary, ensure it's readable
            st.dataframe(df_history[['Timestamp', 'Result', 'Inputs']]) 
            
            # --- Download Raw History Data ---
            # Prepare data for download: flatten 'Inputs' dictionary into separate columns
            df_history_flat = df_history.copy()
            # Convert 'Inputs' column, which stores dicts, into separate columns
            df_history_flat = pd.concat([df_history_flat.drop('Inputs', axis=1), df_history_flat['Inputs'].apply(pd.Series)], axis=1)

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
            df_plot = df_plot.set_index('Timestamp') # Set timestamp as index for plotting

            if disease_to_show == 'Diabetes':
                try:
                    # Safely extract numeric values from 'Inputs' dictionary, coercing errors to NaN and dropping
                    df_plot['Glucose'] = df_plot['Inputs'].apply(lambda x: x.get('Glucose')).apply(pd.to_numeric, errors='coerce')
                    df_plot['BMI'] = df_plot['Inputs'].apply(lambda x: x.get('BMI')).apply(pd.to_numeric, errors='coerce')
                    df_plot['BloodPressure'] = df_plot['Inputs'].apply(lambda x: x.get('BloodPressure')).apply(pd.to_numeric, errors='coerce')

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
                    df_plot['Cholesterol'] = df_plot['Inputs'].apply(lambda x: x.get('chol')).apply(pd.to_numeric, errors='coerce')
                    df_plot['Thalach'] = df_plot['Inputs'].apply(lambda x: x.get('thalach')).apply(pd.to_numeric, errors='coerce')
                    df_plot['RestingBP'] = df_plot['Inputs'].apply(lambda x: x.get('trestbps')).apply(pd.to_numeric, errors='coerce')

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
                    df_plot['HNR'] = df_plot['Inputs'].apply(lambda x: x.get('HNR')).apply(pd.to_numeric, errors='coerce')
                    df_plot['Jitter_percent'] = df_plot['Inputs'].apply(lambda x: x.get('Jitter_percent')).apply(pd.to_numeric, errors='coerce')
                    df_plot['Shimmer'] = df_plot['Inputs'].apply(lambda x: x.get('Shimmer')).apply(pd.to_numeric, errors='coerce')

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
                outcome_counts = df_history['Result'].value_counts()
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
