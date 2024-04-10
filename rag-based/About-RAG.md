*What Is Retrieval Augmented Generation, or RAG?*

Retrieval augmented generation, or RAG, is an architectural approach that can improve the efficacy of large language model (LLM) applications by leveraging custom data. This is done by retrieving data/documents relevant to a question or task and providing them as context for the LLM. RAG has shown success in support chatbots and Q&A systems that need to maintain up-to-date information or access domain-specific knowledge.

Components of a typical RAG architecture

1). Data/ Document Loading: This step will help with loading the relevant data from data sources like websites, pdf, docx, txt files, api's
2). Chunking: This step will break the data into chunks of fixed sizes so they can be processed by the language models<br>
3). Embedding: The chunks created are converted into embeddings, which are high-dimensional vectors that capture the semantic meaning of words, sentences or even entire documents<br>
4). Vectors Stores: These embeddings are then stored in a special database called vector database for future retrieval<br>
5). Querying: When the user queries the chat application, the system will query the vector database and use the large language model to generate the response based on the relevant content it found in the vector database
