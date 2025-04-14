from fastrtc_funasr import get_stt_model
from fastrtc import Stream, ReplyOnPause


stt_model = get_stt_model()



def echo(audio):
    prompt = stt_model.stt(audio)
    print("用户输入:",prompt)
    return prompt

if __name__ == "__main__":
    # stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
    stream = Stream(ReplyOnPause(echo), modality="audio", mode="send")
    stream.ui.launch()