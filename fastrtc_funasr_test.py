from fastrtc_funasr import get_stt_model
from fastrtc import Stream, ReplyOnPause # get_tts_model
from dotenv import load_dotenv
import os
from openai import OpenAI
from kokoro_zh import get_tts_model

# 加载环境变量
load_dotenv()

stt_model = get_stt_model()
tts_model = get_tts_model()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL")
    # api_key='ollama', base_url="http://127.0.0.1:11434/v1"
)


def echo(audio):
    prompt = stt_model.stt(audio)
    print("用户输入:",prompt)
    # print("LLM输出:",prompt)
    response = client.chat.completions.create(
        # model="deepseek-chat",
        model=os.getenv("LLM_MODEL"),
        messages=[
            {"role": "user", "content": prompt+",注意:回答必须为中文"}
        ],
        stream=False
        )
    llm_prompt = response.choices[0].message.content
    print("LLM输出:",llm_prompt)
    for audio_chunk in tts_model.stream_tts_sync(llm_prompt):
        yield audio_chunk
    

if __name__ == "__main__":
    # stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
    stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
    stream.ui.launch()