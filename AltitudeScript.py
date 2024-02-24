import pickle
import sklearn
import pandas as pd
import langchain

from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
import sys
print("Python Version:", sys.version)
print("Python Executable:", sys.executable)


# Load the model
with open('/Users/srinivas/AltitudeApp/knn_model.pkl', 'rb') as file:
    pipeline_regressor = pickle.load(file)



def get_langchain_response(input_data):
    # Initialize LangChain components as needed
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="INSERT_KEY")

    # Construct the prompt from input_data
    prompt = construct_prompt(input_data)

    # Invoke the chain and get response using the updated method
    try:
        response = llm.invoke(prompt)  # Replacing __call__ with invoke
    except Exception as e:
        print("Error during invocation:", e)
        response = "Error occurred in processing."
    # Assuming the response text is in an attribute named 'text'
    response_text = response.text if hasattr(response, 'text') else str(response)

    # Replace newline characters with HTML line break tags
    response_text = response_text.replace("\n", "<br>")

    return response_text



def get_prediction(new_data):
    # Assuming new_data is a DataFrame with the required structure
    # Ensure the DataFrame structure matches the input expected by the regressor model
    processed_input = pipeline_regressor.named_steps['preprocessor'].transform(new_data)

    # Predict the LLS score using the regression model
    predicted_lls_score = pipeline_regressor.named_steps['grid_search'].best_estimator_.predict(processed_input)

    # The predicted LLS score is returned as an array; extract the first element
    lls_score = predicted_lls_score[0]

    # Return the predicted LLS score; no need to scale as it's already in the correct range (0 to 12)
    return {'predicted_lls_score': lls_score}

def construct_prompt(input_data):
    # Create a formatted prompt from the input data
    prompt = "Provide a medical analysis and suggestions for the following patient profile:\n"
    prompt += f"BVP: {input_data['BVP']}\n"
    prompt += f"EDA: {input_data['EDA']}\n"
    prompt += f"Theta: {input_data['Theta']}\n"
    prompt += f"Alpha1 {input_data['Alpha1']}\n"
    prompt += f"Beta1 {input_data['Beta1']}\n"
    prompt += f"Age: {input_data['age']}\n"
    prompt += f"Gender: {input_data['gender']}\n"
    prompt += f"Temperature: {input_data['temp']}°F\n"
    prompt += f"Systolic Blood Pressure: {input_data['bp_systolic']} mmHg\n"
    prompt += f"Diastolic Blood Pressure: {input_data['bp_diastolic']} mmHg\n"
    prompt += f"Blood Oxygen Saturation: {input_data['spo2']}%\n"
    prompt += f"Pulse Rate: {input_data['pulse']} bpm\n"
    prompt += "Hypertension: {}\n".format("Yes" if input_data['hypertension'] else "No")
    prompt += "Diabetes: {}\n".format("Yes" if input_data['diabetes'] else "No")
    prompt += "Smoking: {}\n".format("Yes" if input_data['smoking'] else "No")
    prompt += "Symptoms: "
    prompt += "Headache, " if input_data['sym_headache'] else ""
    prompt += "Gastrointestinal Distress, " if input_data['sym_gi'] else ""
    prompt += "Fatigue, " if input_data['sym_fatigue'] else ""
    prompt += "Dizziness" if input_data['sym_dizziness'] else ""
    prompt = prompt.rstrip(', ')  # Remove the last comma
    prompt += "\nPermanent Altitude: {} meters\n".format(input_data['permanent_altitude'])
    prompt += "Altitude Gain From: {} meters\n".format(input_data['alt_gain_from_altitude'])
    prompt += "Altitude Gain To: {} meters\n".format(input_data['alt_gain_to_altitude'])
    prompt += f"Ascent Day: {input_data['ascent_day']}\n"
    prompt += f"LLS Score: {input_data['lls_score']}\n"
    prompt += "\nPlease provide an assessment and recommendations based on the above profile."

    return prompt



    
# Update the sample input data with new fields
sample_data = {
    'age': [47],  # Age of the individual
    'gender': ['F'],  # Gender: 'M' for male, 'F' for female
    'temp': [97],  # Temperature in degrees Celsius (normal body temperature)
    'bp_systolic': [100],  # Systolic blood pressure (normal range)
    'bp_diastolic': [110],  # Diastolic blood pressure (normal range)
    'spo2': [98],  # Blood oxygen saturation in percentage (normal range)
    'pulse': [90],  # Pulse rate in beats per minute (normal range)
    'hypertension': [1],  # 0 if the individual does not have hypertension
    'diabetes': [1],  # 0 if the individual does not have diabetes
    'permanent_altitude': [6000],  # Altitude of permanent residence in meters (example)
    'alt_gain_from_altitude': [800],  # Altitude gain from starting point (example)
    'alt_gain_to_altitude': [3800],  # Altitude gain to end point (example)
    'ascent_day': [1],  # Number of days spent ascending to current altitude
    'smoking': [1],  # 0 if the individual does not smoke
    'sym_headache': [0],  # 0 if not experiencing headache
    'sym_gi': [0],  # 0 if not experiencing gastrointestinal distress
    'sym_fatigue': [0],  # 0 if not experiencing fatigue
    'sym_dizziness': [1],  # 0 if not experiencing dizziness
    'BVP': [-0.15],  # Blood Volume Pulse (example value)
    'EDA': [0.02],  # Electrodermal Activity (example value)
    'Theta': [0.02], 
    'Alpha1': [0.01], 
    'Beta1': [0.02],
}

def test_construct_prompt():
    # Sample input data similar to what you expect from the Shiny app
    sample_input = {
    'age': [47],  # Age of the individual
    'gender': ['F'],  # Gender: 'M' for male, 'F' for female
    'temp': [97],  # Temperature in degrees Celsius (normal body temperature)
    'bp_systolic': [100],  # Systolic blood pressure (normal range)
    'bp_diastolic': [110],  # Diastolic blood pressure (normal range)
    'spo2': [98],  # Blood oxygen saturation in percentage (normal range)
    'pulse': [90],  # Pulse rate in beats per minute (normal range)
    'hypertension': [1],  # 0 if the individual does not have hypertension
    'diabetes': [1],  # 0 if the individual does not have diabetes
    'permanent_altitude': [6000],  # Altitude of permanent residence in meters (example)
    'alt_gain_from_altitude': [800],  # Altitude gain from starting point (example)
    'alt_gain_to_altitude': [3800],  # Altitude gain to end point (example)
    'ascent_day': [1],  # Number of days spent ascending to current altitude
    'smoking': [1],  # 0 if the individual does not smoke
    'sym_headache': [0],  # 0 if not experiencing headache
    'sym_gi': [0],  # 0 if not experiencing gastrointestinal distress
    'sym_fatigue': [0],  # 0 if not experiencing fatigue
    'sym_dizziness': [1],  # 0 if not experiencing dizziness
    'BVP': [-0.15],  # Blood Volume Pulse (example value)
    'EDA': [0.02],  # Electrodermal Activity (example value)
    'Theta': [0.02], 
    'Alpha1': [0.01], 
    'Beta1': [0.02],
    'lls_score': [3]
}

    # Call the construct_prompt function with the sample input
    prompt = construct_prompt(sample_input)

    # Print the result to verify correctness
    print("Constructed Prompt:", prompt)
    
def test_chatbot():
    # Example test input, structured as you expect from your Shiny app
    test_input = {
        'age': 50,
        'gender': 'M',
        'permanent_altitude': 100,
        'bp_systolic': 135,
        'bp_diastolic': 85,
        'spo2': 98,
        'pulse': 70,
        'hypertension': 1,
        'diabetes': 0,
        'ascent_day': 1,
        'smoking': 0,
        'sym_headache': 0,
        'sym_gi': 0,
        'sym_fatigue': 1,
        'sym_dizziness': 0,
        'alt_gain_from_altitude': 500,
        'alt_gain_to_altitude': 1000,
        'temp': 98,
        'risk_percentage': 20,
        'BVP': [-0.15],  # Blood Volume Pulse (example value)
        'EDA': [0.02],  # Electrodermal Activity (example value)
        'Theta': [0.02], 
        'Alpha1': [0.01], 
        'Beta1': [0.02],
        'lls_score': [3]
    }

    # Call the get_langchain_response function
    response = get_langchain_response(test_input)
    
    # Print the response to see the output
    print("Response from Google AI:", response)

sample_input_df = pd.DataFrame(sample_data)
test_chatbot()

# # # Test the get_prediction function
test_construct_prompt()

result = get_prediction(sample_input_df)
print(result)