from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

if __name__ == '__main__':
    query = "What is a stock?"
    print(query)
    messages = [
        ("system", "You are a helpful assistant that answers questions"),
        ("human", f"{query}."),
    ]

    answer = llm.invoke(messages)
    print(answer.content)
