from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


def create_embeddings(chunks, category=None):
    embedding_function = OpenAIEmbeddings()
    # embedding_function.model = 'text-embedding-3-small'
    if category == 'StockInfo':
        vector_store = Chroma.from_documents(chunks, embedding_function, persist_directory='./stock_info_db')
    else:
        vector_store = Chroma.from_documents(chunks, embedding_function, persist_directory='./general_info_db')
    return vector_store


def get_db(category=None):
    embedding_function = OpenAIEmbeddings()
    if category == 'StockInfo':
        return Chroma(persist_directory="./stock_info_db", embedding_function=embedding_function)
    else:
        return Chroma(persist_directory="./general_info_db", embedding_function=embedding_function)


def ask_and_get_answer(vector_store, q, k=3):
    # wrapped_query =
    llm = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer
