'''
Description: 
Author: Jingjing Li
Date: 2023-09-10 16:26:32
LastEditors: Jingjing Li
LastEditTime: 2023-09-15 15:52:06
'''
import openai
import pandas as pd
import numpy as np
import os
import json
from typing import Tuple, Optional


class AiAgent():
    def __init__(self,model = "gpt-3.5-turbo",temp=0): #name
        self.model = model
        self.messages = []
        self.temp = temp
    def create_messages(self,text):
        """
        Initialize 'messages' as empty
        Add system message
        """
        self.messages = []
        self.messages.append({"role": "system", "content": f"{text}"})
        return self #because sometimes use default __init__ funct
    def add_user(self,text):
        """
        Add user input(question)
        """
        self.messages.append({"role": "user", "content": f"{text}"})
    def add_assistant(self,text):
        """
        Add assistant response
        """
        self.messages.append({"role": "assistant", "content": f"{text}"})
    def run(self):
        """
        Combine 'ChatCompletion' + Add response to messages
        """
        resp = self.openai_search(self.messages,self.model,self.temp) #gpt-4 #gpt-3.5-turbo
        self.add_assistant(resp)
        return resp
    def run_ai_search(self):
        """
        Run 'ChatCompletion', no memory
        """
        return self.openai_search(self.messages,self.model,self.temp)
    
    
    def openai_search(self,messages,model_name,temp):
        out = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=temp, # this is the degree of randomness of the model's output
        )
        return out['choices'][0]['message']['content']
    

def convert_to_json(code_resp: str, format_syst: Optional[str] = None) -> Tuple[AiAgent, str]:
    """
    Convert the given code response to JSON format using an AI agent.

    Parameters:
    code_resp (str): LLM output from DS agent
    format_syst (str): System message for formatter agent

    Returns:
    Tuple[AiAgent, str]: The AI agent and the response in JSON format
    """
    format_syst = format_syst or "You are good at following output format instructions in user message strictly. Don't write extra content other than specified."
    format_prompt = f"""Extract Python code and explanation from the given string. Construct response in JSON format: 
{{"code":[generated code for ML model],
"explanation": [simple explanation]}}

String: {code_resp}
"""
    while True:
        format_agent = AiAgent(model="gpt-4")
        format_agent.create_messages(format_syst)
        format_agent.add_user(format_prompt)
        resp_json = format_agent.run()

        try:
            json.loads(resp_json)
            return format_agent, resp_json
        except json.JSONDecodeError:
            continue