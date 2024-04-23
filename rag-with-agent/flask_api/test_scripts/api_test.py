from langchain.chains.api.base import APIChain

from langchain.chains.api import open_meteo_docs
from langchain_openai import OpenAI

from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


llm = OpenAI(temperature=0)

prompt_template = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question using the API response: {question}"
)

# Reference: https://python.langchain.com/docs/modules/chains/
# Reference: https://python.langchain.com/docs/use_cases/apis/#going-deeper
api_chain = APIChain.from_llm_and_api_docs(
    llm=llm,
    api_docs=open_meteo_docs.OPEN_METEO_DOCS,
    prompt=prompt_template,
    limit_to_domains=["https://api.open-meteo.com/"]
)


# Use the APIChain to make a prediction
inputs = {"question": "What is the weather like right now in Wayne in degrees celsius?"}
output = api_chain.invoke(inputs)

print("API Test")
print(output)

