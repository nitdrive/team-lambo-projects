from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="Based on the recent stock performance of Meta Platforms Inc. (META), the stock price has fluctuated over the past 5 days."
)

response.stream_to_file(speech_file_path)