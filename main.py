import warnings
warnings.filterwarnings("ignore")
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.chat_models import ChatOpenAI
import os

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#image2text
def img2text(image):
    image2text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text = image2text(image)[0]["generated_text"]

    print(text)
    return text



#llm
def generate_story(scenario):
    template = """
    You are a storyteller;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(
        llm=ChatOpenAI(
            model="gpt-3.5-turbo", temperature=1
        ), 
        prompt=prompt, verbose=True
    )

    story = story_llm.predict(scenario=scenario)
    print(story)
    return story


# Text2Speech

def text2speech(story):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi/ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": story
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)

scenario = img2text("haha.jpg")
story = generate_story(scenario)
text2speech(story)
