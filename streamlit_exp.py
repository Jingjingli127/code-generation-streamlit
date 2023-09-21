'''
Description: 
Author: Jingjing Li
Date: 2023-09-08 10:32:53
LastEditors: Jingjing Li
LastEditTime: 2023-09-18 17:25:18
'''
import streamlit as st
import openai
import os  
import numpy as np
import pandas as pd
from io import StringIO
import json
import time
import argparse
from openai_llm import *


# Define Streamlit app layout
# MUST be the first Streamlit command on an app page; only be set once per page
## page_title: default to filename of script
st.set_page_config(
    page_title = "Code Generation Using GPT Demo",
    layout = 'wide'
)
st.title("Code Generation Using GPT-4")


# Set up OpenAI API credentials
# Use Option # 1 - Using Streamlit secrets
openai.api_key = st.secrets['api_secret']


# Data input
def design_input_st():
    """
    Design data input UI
    1. Allow user to pass in data string, or upload a data file
    2. Convert into pd.DataFrame
    """
    ## either data string/dataframe
    df = None
    # Using "with" notation
    with st.sidebar:
        st.write("## Enter sample/full data to LLM :ice_cube:")
        data_option = st.radio(
            "**Choose a way to input data:**",
            ["Manually enter data", "Upload file"],
            captions = ["Requirement\: comma separated, no quotes, first row is column names",
                        "Accept .csv/xlsx/json file"]
        )
        if data_option == "Manually enter data":
            data_string = st.text_area("Enter data input to LLM") #return str; if enter nothing, still return ''
            if data_string: #''--False
                #sample data in json
                if (data_string.startswith('{')) and (data_string.endswith('}')): 
                    df = json.loads(data_string)
                #sample data in csv
                else:
                    # Create a StringIO object from the string-data (comma separated)
                    csv_io = StringIO(data_string)

                    # Read the CSV string into a DataFrame
                    df = pd.read_csv(csv_io) #df either comes from here/param
        elif data_option == "Upload file":
            uploaded_file = st.file_uploader("Choose a file to upload",type=['csv','xlsx','json'],accept_multiple_files=False)
            if uploaded_file: # None or UploadedFile object
                extension = uploaded_file.name.split(".")[-1] #uploaded_file.name: e.g:'train.csv'
                if extension == 'csv':
                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                elif extension == 'xlsx':
                    df = pd.read_excel(uploaded_file)
                elif extension == 'json':
                    # method1
                    df = pd.read_json(uploaded_file,lines=True) #data written in lines sep by '\n' endlines

    return df #None or actual object


def display_data(df):
    """
    Display data on UI
    """
    st.header("Data:")
    if df is None: #no input data
        st.markdown("**No input data given to LLM. LLM will generate code following general framework**")
        return 
    else: #input data
        is_check = st.toggle('See dataframe') #Whether or not the toggle is checked
        if is_check:
            # JSON
            if isinstance(df,dict):
                st.write(df)
            # DataFrame
            elif isinstance(df,pd.DataFrame):
                st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
                st.write("**First five rows of input data:**")
                st.dataframe(data=df.head())
        return df       

def add_data(ds_agent,data):
    """
    Add given data into LLM's conversation history
    ds_agent: AiAgent object
    data: pd.DataFrame
    return: None
    """
    if data is not None:
        # print("Data passed into LLM:",data)
        dataset_desc = f"""Given dataset: 
        {data}. Wait for data science questions to be given.
"""
        ds_agent.add_user(dataset_desc)
    return ds_agent

def design_model_st():
    """
    Design model setting UI
    """
    with st.sidebar:
        st.write("## Choose LLM settings :gear:")
        ds_model = st.radio(
            "**Choose a model version:**",
            ["gpt-4", "gpt-3.5-turbo", "Other"],
            captions = ["Current version: 06/13",
                        "Model behind ChatGPT, current version: 06/13"
                        ]
        )
        if ds_model == 'Other':
            ds_model = st.text_input('Enter a valid model version', placeholder = 'Enter a valid model version...',
                                     label_visibility='collapsed')
            if len(ds_model) > 0:
                try:
                    openai.ChatCompletion.create(
                        model=ds_model,
                    messages = [{"role": "system", "content": "You are a helpful assistant."}]) #random messages, since 'messages' cannot be empty
                except Exception as e:
                    # display exception
                    st.exception(e)
                    return None, None
        temp = st.slider(label="**Choose Temperature parameter** :unicorn_face:",
                         help = 'Control creativity of the generated response. Bigger value means more creative',
                  min_value = 0.0,
                  max_value = 1.0,
                  value = 0.0, #default to min_value
                  step = 0.1,
                  format = '%.1f')
    return ds_model, temp
 





def design_prompt(prompt_lookup,q_list=None):
    """
    prompt_lookup: dictionary, lookup table for more detailed prompts after testing
    q_list: None or list, contain example queries allowing user to choose from
    """
    if not q_list:
        q_list = ['Build machine learning model',
                    'Draw graphs and output statistics for EDA',
                    'Tune hyperparameter',
                    'Other']
    st.header("Query:")
    with st.chat_message("user"):
        q = st.selectbox('Select an example query:', q_list)
        # APP logic
        ## get detailed prompt--prompt that works after testing
        if q in prompt_lookup:
            q = prompt_lookup[q]
        if q == "Other":
            q = st.text_input('Enter your query:', placeholder = 'Enter query here ...')
    prompt = f"""Write Python code to answer the following question. \
Question: {q} 
Output code and simple explanation.
"""
    return prompt



def generate_response(ds_agent,prompt):
    ds_agent.add_user(prompt)
    resp = ds_agent.run() #chat + add
    return ds_agent,resp
    
def convert_to_json(code_resp,format_syst=None):
    """
    code_resp: string, LLM output from DS agent
    format_syst: string, system message for formatter agent
    """
    format_syst = format_syst or "You are good at following output format instructions in user message strictly. Don't write extra content other than specified."
    # prompt design
    format_prompt = f"""Extract code and explanation from the given string. \
Construct response in JSON format: 
{{"code":[generated code from given string],
"explanation": [simple explanation]}}

String: {code_resp}
"""
    while True:
        format_agent = AiAgent(model="gpt-4").create_messages(format_syst)
        format_agent.add_user(format_prompt)
        resp_json = format_agent.run() #chat + add
        try:
            json.loads(resp_json)
            return format_agent,resp_json
        except json.decoder.JSONDecodeError:
            continue

def extract_elements(resp_json):
    """
    After converting response from LLM to JSON format,
    extract needed elements:
    'code', 'explanation'
    """
    code = json.loads(resp_json)['code']
    explanation = json.loads(resp_json)['explanation']
    return code, explanation
    

    
def main(ds_syst,format_syst):
    """
    ds_syst: optional system message for data scientist agent
    format_syst: optional system message for formatter agent
    """
    # data input in side bar
    df = design_input_st()

    # display data on main page
    data = display_data(df)

    # model settings in side bar
    ds_model, temp = design_model_st()
    if ds_model is None:
        return ## invalid GPT model
            
    # Design prompt
    prompt_lookup = {
        "Build machine learning model":"""Write python code to build machine learning model to predict target variable. \
If given sample data, pay attention to format and details in sample data. \
Code follow these requirements:
1. Wrap each functionality using functions as much as possible; include a final main() function.
2. Allow passing parameters from command line: 'data path', 'target name', 'target type'.
3. Evaluation function depends on 'target type'."""
    }
    prompt = design_prompt(prompt_lookup)

    # create LLM agents
    ds_agent = AiAgent(model=ds_model,temp=temp).create_messages(ds_syst)
    ds_agent = add_data(ds_agent,data)

    # chat with LLM
    if st.button("Chat!",type='primary'):
        with st.spinner(f'{ds_model.upper()} running. Please wait...'):
            start_time = time.time()
            ds_agent,resp_llm = generate_response(ds_agent,prompt)
            
            # convert to JSON
            _,resp_json = convert_to_json(resp_llm,format_syst)
            end_time = time.time()
            st.success(f'Chat complete in {(end_time - start_time):.0f}s!')

        # extract elements
        code, explanation = extract_elements(resp_json)
        print(code)
        st.header(f"Code from {ds_model.upper()}:")
        # download in txt file
        st.download_button('Download code',code)
        with st.chat_message("assistant"):
            st.code(code, language="python", line_numbers=False)

        st.header("Explanation:")
        with st.chat_message("assistant"):
            st.write(explanation)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_syst', 
                        type = str, 
                        default = "You are a senior data scientist who is good at analyzing data and building models.",
                        help = "System message for data scientist agent"
    )

    parser.add_argument('--format_syst', 
                        type = str, 
                        default = "You are good at following output format instructions in user message strictly. Don't write extra content other than specified.",
                        help = 'System message for formatter agent')
    
    args = parser.parse_args()
    main(args.ds_syst,args.format_syst)



    
# other queries could be:
## Data cleaning on 'Name' column using NLP techniques
## Write a regular expression that split text into sentences. Make sure edge cases like 'J. K. Rowling', 'e.g.' are not splitted.
## Tune hyperparameter using bayesian method
# could also apply to function-level coding:
## no data + query


# Define function to explain code using OpenAI Codex
# def explain_code(input_code, language):
#     model_engine = "gpt-4" # Change to the desired OpenAI model
#     prompt = f"Explain the following {language} code: \n\n{input_code}"
#     response = openai.ChatCompletion.create(
#         model = model_engine,
#         messages = [
#             {"role": "user", "content": f"{prompt}"}
#         ],
#         temperature=0,
#     )
#     return response.choices[0].message["content"]

# # Temperature and token slider
# temperature = st.sidebar.slider(
#     "Temperature",
#     min_value=0.0,
#     max_value=1.0,
#     value=0.5,
#     step=0.1
# )
# tokens = st.sidebar.slider(
#     "Tokens",
#     min_value=64,
#     max_value=2048,
#     value=256,
#     step=64
# )
# Define Streamlit app behavior
# if st.button("Explain"):
#     st.header("Code:")
#     st.code(code_input, language="python", line_numbers=False)
#     output_text = explain_code(code_input, language)
#     st.header("Code Explanation")
#     st.text_area("Code Explanation",output_text)

#     st.balloons()

# def ds_model_version(model_list=None):
#     """
#     Let user determine which model version to use
#     model_list: list of possible choices, default=None (no need to give)
#     If 'other' is chosen, user need to enter the model they want to use
#     """
#     if not model_list:
#         model_list = ['gpt-3.5-turbo', 'gpt-4', 'Other']
#     with st.chat_message("user"):
#         ds_model = st.selectbox('Select a model you want to use:', model_list)
#         # APP logic
#         if ds_model == 'Other':
#             ds_model = st.text_input('Enter the model version:', placeholder = 'Enter model version here ...')
#     return ds_model


