import chainlit as cl
import os
import re  
from typing import cast
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

huggingfacehub_api_token = os.getenv('API_KEY')

huggingface_llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    huggingfacehub_api_token=huggingfacehub_api_token
)

previous_messages = []
MAX_HISTORY = 5  

@cl.on_chat_start
async def on_chat_start():
    model = ChatHuggingFace(streaming=True, llm=huggingface_llm)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an AInstein, All-round AI Assistant. You are a friendly, direct-response AI chatbot designed to assist with a variety of tasks, making the user experience smooth and easy. You are here to provide simple, welcoming, and direct answers. 
            ***Respond with simple, welcoming, and direct answers only.***"""),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))
    msg = cl.Message(content="")

    previous_messages.append(f"User: {message.content}")

    if len(previous_messages) > MAX_HISTORY:
        previous_messages.pop(0)

    combined_context = "\n".join(previous_messages)

    print(f"Combined context: {combined_context}")

    try:
        response_content = ""

        async for chunk in runnable.astream(
            {"question": combined_context},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            response_content += chunk

        cleaned_response = re.sub(r"<(think|\/think)>.*?</think>", "", response_content, flags=re.DOTALL).strip()  
        cleaned_response = re.sub(r"^(I should|Alright|Okay|Sure|I see|The user said|You mentioned|I should respond).*?\.", "", cleaned_response, flags=re.MULTILINE).strip()

        def remove_before_substring(text, substring):
            index = text.find(substring)
            if index != -1:
                return text[index + len(substring):].strip()
            return text 
        
        cleaned_response = remove_before_substring(cleaned_response, "</think>")

        if cleaned_response and not cleaned_response.isspace():
            previous_messages.append(f"Assistant: {cleaned_response}")  
            msg.content = cleaned_response  
            await msg.send()  
            print("Response sent successfully.")
        else:
            await msg.send("Error: No valid response generated.")

    except ValueError as ve:
        msg.content = f"A protocol error occurred: {ve}"
        await msg.send()
        print(f"ValueError: {ve}")
    except Exception as e:
        msg.content = f"An error occurred: {e}"
        await msg.send()
        print(f"Error occurred: {e}")
