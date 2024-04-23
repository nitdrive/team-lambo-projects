from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunker:
    @classmethod
    def chunk_data(cls, data, chunk_size=256, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(data)
        return chunks
