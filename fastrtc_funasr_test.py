from fastrtc_funasr import get_stt_model
from fastrtc import Stream, ReplyOnPause,get_tts_model
from dotenv import load_dotenv
import os
from openai import OpenAI

# 加载环境变量
load_dotenv()

stt_model = get_stt_model()
tts_model = get_tts_model()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


def echo(audio):
    prompt = stt_model.stt(audio)
    print("用户输入:",prompt)
    # print("LLM输出:",prompt)
    # response = client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ],
    #     stream=False
    #     )
    # llm_prompt = response.choices[0].message.content
    # print("LLM输出:",llm_prompt)
    for audio_chunk in tts_model.stream_tts_sync(prompt):
        yield audio_chunk
    

if __name__ == "__main__":
    # stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
    stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
    stream.ui.launch()