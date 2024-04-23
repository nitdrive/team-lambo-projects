import tiktoken


# Choose the right model based on pricing https://openai.com/pricing, find the price per 1000 tokens
def calculate_embedding_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, (total_tokens * 0.00002) / 1000
