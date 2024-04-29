from langchain_core.documents import Document


class MetadataEnhancer:
    @classmethod
    def add_metadata_to_documents(cls, docs: list[Document], metadata: dict):
        for doc in docs:
            current_metadata = doc.metadata
            for key in metadata:
                if key not in current_metadata:
                    current_metadata[key] = metadata[key]
                    doc.metadata = current_metadata

        return docs
