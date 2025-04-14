from fastrtc import Stream, ReplyOnPause,get_stt_model, get_tts_model
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

stt_model = get_stt_model()
tts_model = get_tts_model()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

# def echo(audio: tuple[int, np.ndarray]):
#     # The function will be passed the audio until the user pauses
#     # Implement any iterator that yields audio
#     # See "LLM Voice Chat" for a more complete example
#     yield audio

def echo(audio):
    prompt = stt_model.stt(audio)
    print("用户输入:",prompt)
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": prompt}
    ],
    stream=False
    )
    prompt = response.choices[0].message.content
    print("LLM输出:",prompt)
    for audio_chunk in tts_model.stream_tts_sync(prompt):
        yield audio_chunk

# stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")

if __name__ == "__main__":
    # stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
    stream = Stream(ReplyOnPause(echo), modality="audio", mode="send")
    stream.ui.launch()
