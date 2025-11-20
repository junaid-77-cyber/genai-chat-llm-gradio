## Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

### AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### PROBLEM STATEMENT:
To develop a web-based AI chatbot using the FalconLM-Instruct model with streaming responses and customizable system instructions using Gradio UI and the text_generation API.
### DESIGN STEPS:

#### Step 1: Import the necessary libraries
Import os, dotenv, text_generation, gradio, and other required packages for API connection and UI.

#### Step 2: Load API key from the .env file
Use load_dotenv() to read the API key securely from environment variables.

#### Step 3: Initialize Falcon Client connection
Create a Client instance using the Hugging Face endpoint and pass authentication headers.

#### Step 4: Format the chat prompt
Define a function to merge system message, user input, and previous chat history into a structured format.

#### Step 5: Generate AI response using streaming
Use generate_stream() to produce token-by-token output in real-time and append it into chat history.

#### Step 6: Create Gradio user interface
Design a UI with Chatbot area, Prompt input, System instruction box, Temperature slider, and Submit/Clear buttons using gr.Blocks().

#### Step 7: Launch the chatbot application
Enable queue support for streaming and run the final interface using demo.queue().launch().

### PROGRAM:
```
import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
# Helper function
import requests, json
from text_generation import Client

#FalcomLM-instruct endpoint on the text_generation library
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Basic {hf_api_key}"}, timeout=120)
def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt
def respond(message, chat_history, instruction, temperature=0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = client.generate_stream(prompt,
                                      max_new_tokens=1024,
                                      stop_sequences=["\nUser:", "<|endoftext|>"],
                                      temperature=temperature)
                                      #stop_sequences to not generate the user answer
    acc_text = ""
    #Streaming the tokens
    for idx, response in enumerate(stream):
            text_token = response.token.text

            if response.details:
                return

            if idx == 0 and text_token.startswith(" "):
                text_token = text_token[1:]

            acc_text += text_token
            last_turn = list(chat_history.pop(-1))
            last_turn[-1] += acc_text
            chat_history = chat_history + [last_turn]
            yield "", chat_history
            acc_text = ""
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    with gr.Accordion(label="Advanced options",open=False):
        system = gr.Textbox(label="System message", lines=2, value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.")
        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.7, step=0.1)
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot]) #Press enter to submit


demo.queue().launch()    
```

### OUTPUT:
![alt text](<Screenshot 2025-11-21 052038.png>)
![alt text](<Screenshot 2025-11-21 052006.png>)

### RESULT:
An interactive LLM-powered chatbot was successfully created. The chatbot accepts user prompts, streams responses in real time, and allows control over system instructions and temperature through a Gradio interface.