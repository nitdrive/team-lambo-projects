import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader


class DocumentLoader:
    @classmethod
    # loading PDF, DOCX and TXT files as LangChain Documents
    def load_document(cls, file):
        name, extension = os.path.splitext(file)
        print(f'Loading {file}')
        if extension == '.pdf':
            loader = PyPDFLoader(file)
        elif extension == '.docx':
            loader = Docx2txtLoader(file)
        elif extension == '.txt':
            loader = TextLoader(file)
        else:
            print('Document format is not supported!')
            return None

        data = loader.load()
        return data
