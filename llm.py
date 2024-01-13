import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI()
OpenAI.api_key =os.getenv('OPENAI_API_KEY')
from config import standard_size
def pass_to_llm(document, model="gpt-3.5-turbo-instruct"):
    response = client.completions.create(
        model=model,
        prompt=document,
        max_tokens=standard_size
    )
    if 'choices' in response and response['choices']:
        content = response['choices'][0]['message']['content']
        return content
    else:
        print("Error in API response:")
        print(response)
        return None
    pass
