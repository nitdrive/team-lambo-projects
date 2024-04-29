import pandas as pd
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def memory_usage() -> pd.DataFrame:
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    return pd.DataFrame(
        [
            {
                "gpu": gpu_stats.name,
                "max_memory_gb": max_memory,
                "used_memory_gb": start_gpu_memory,
            }
        ]
    )


def model_billion_parameters(model, round_digits: int = 2) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    return round(n_params / 1e9, round_digits)


MODEL_NAME = "havenhq/mamba-chat"

ANSWER_START = "<|assistant|>\n"
ANSWER_END = "<|endoftext|>"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.eos_token = ANSWER_END
tokenizer.pad_token = tokenizer.eos_token

# print(AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5').chat_template)
# print(AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template)
tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

# print(tokenizer.chat_template)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = MambaLMHeadModel.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float16)

# print(DEVICE)

# print(f"{model_billion_parameters(model)}B parameters")

# print(memory_usage().to_string())


# messages = [{"role": "user", "content": prompt}]

run = True
while run:
    prompt = input("What is your question?\n")
    print(f"Q: {prompt}")
    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True).to(DEVICE)

    # print(input_ids)

    # print(tokenizer)

    outputs = model.generate(
        input_ids=input_ids,
        max_length=1024,
        temperature=0.9,
        top_p=0.7,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0])
    print(response)
    should_run = input("Do you want to ask another question? (Y/N)")
    if should_run != 'Y':
        run = False
