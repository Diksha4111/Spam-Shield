import json
from flask import requests
import streamlit as st

# Content Filtration App
st.title("Text Classification App")

# User input for text
user_input = st.text_input("Enter text:")

# Button to trigger classification by calling content filtration API
if st.button("Classify"):

    # Integrating content filtration API in the app
    url = f"http://127.0.0.1:8080/classify_api?input_text={user_input}"
    payload = ""
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    print(response.text)

    # Load response data in json format 
    response_data = json.loads(response.text)

    # Extracting values from response data
    prediction_value = response_data.get('prediction')
    accuracy_value = response_data.get('accuracy')

    # Display the result
    st.markdown(f"Result: {prediction_value}")
    st.write(f"Text: {user_input}")
    st.write(f"Accuracy of classification: {accuracy_value}")


# Button to trigger to check API status
if st.button("Check API Status"):

    # Calling content filtration API 
    url = f"http://127.0.0.1:8080/classify_api?input_text={user_input}"
    payload = ""
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
         
    # Load response data in json format 
    response_data = json.loads(response.text)

    # Extracting values from response data
    prediction_value = response_data.get('prediction')
    accuracy_value = response_data.get('accuracy')  
    
    # Display API status in json format
    r = {'code': 200,
        'text': user_input, 
        'running': True,
        'prediction': prediction_value,
        'accuracy': accuracy_value} 
    st.write(r)

    