from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import click
import librosa
import numpy as np
from numpy.typing import NDArray
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

import torch

curr_dir = Path(__file__).parent.parent
# 模型文件路径
MODEL_DIR = curr_dir / "Model" / "SenseVoiceSmall"
VAD_MODEL_DIR = curr_dir / "Model" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"

STT_MODELS = Literal["SenseVoiceSmall", "paraformer-zh-streaming"]


class STTModel(Protocol):
    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str: ...


class FunasrSTT(STTModel):
    MODEL_OPTIONS = Literal["SenseVoiceSmall", "paraformer-zh-streaming"]
    LANG_OPTIONS = Literal[
            "zh",
            "yue",
            "en", 
            "ja", 
            "ko"
        ]
    def __init__(self, model: STT_MODELS = "SenseVoiceSmall", lang: LANG_OPTIONS = "zh"):
        self.lang = lang
        self.model = AutoModel(
                                model=MODEL_DIR,
                                #vad_model=VAD_MODEL_DIR,
                                vad_kwargs={"max_single_segment_time": 30000},
                                device="cuda:0",
                            )
        # 移除 tokenizer 相关代码
        # self.tokenizer = load_tokenizer()

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sr, audio_np = audio  # type: ignore
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        if sr != 16000:
            audio_np = librosa.resample(
                audio_np, orig_sr=sr, target_sr=16000
            )
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)

        try:
            """
            url: https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
            基于ModelScope进行推理
                推理支持音频格式如下：
                - wav文件路径，例如：data/test/audios/asr_example.wav
                - pcm文件路径，例如：data/test/audios/asr_example.pcm
                - wav文件url，例如：https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav
                - wav二进制数据，格式bytes，例如：用户直接从文件里读出bytes数据或者是麦克风录出bytes数据。
                - 已解析的audio音频，例如：audio, rate = soundfile.read("asr_example_zh.wav")，类型为numpy.ndarray或者torch.Tensor。
                - wav.scp文件，
                    ```
                    cat wav.scp
                    asr_example1  data/test/audios/asr_example1.wav
                    asr_example2  data/test/audios/asr_example2.wav
                    ...

            """
            audio_tensor = torch.from_numpy(audio_np)
            res = self.model.generate(
                    input=audio_tensor,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                )
                
            # 检查结果格式
            if not res or not isinstance(res, list) or len(res) == 0:
                print(click.style("WARN", fg="yellow") + ":\t  No transcription result")
                return ""
            
            if not isinstance(res[0], dict) or "text" not in res[0]:
                print(click.style("WARN", fg="yellow") + ":\t  Invalid transcription format")
                return ""
            
            text = rich_transcription_postprocess(res[0]["text"])
            return text if text else ""
                
        except Exception as e:
            print(click.style("ERROR", fg="red") + f":\t  Transcription failed: {str(e)}")
            return ""



@lru_cache
def get_stt_model(
    model: STT_MODELS = "SenseVoiceSmall", lang: FunasrSTT.LANG_OPTIONS = "zh" 
):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    m = FunasrSTT(model)
    # audio = load_audio(str(curr_dir / "test_file.wav"))
    # print(click.style("INFO", fg="green") + ":\t  Warming up STT model.")

    # m.stt((16000, audio))
    print(click.style("INFO", fg="green") + ":\t  STT model warmed up.")
    return m

# def stt_for_chunks(
#     stt_model: STTModel,
#     audio: tuple[int, NDArray[np.int16 | np.float32]],
#     chunks: list[AudioChunk],
# ) -> str:
#     sr, audio_np = audio
#     return " ".join(
#         [
#             stt_model.stt((sr, audio_np[chunk["start"] : chunk["end"]]))
#             for chunk in chunks
#         ]
#     )

# 添加 AudioChunk 类型定义
class AudioChunk(TypedDict):
    start: int
    end: int

def stt_for_chunks(
    stt_model: STTModel,
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chunks: list[AudioChunk],
) -> str:
    sr, audio_np = audio
    
    # FunASR 流式处理参数
    chunk_size = [0, 10, 5]  # 600ms
    encoder_chunk_look_back = 4
    decoder_chunk_look_back = 1
    chunk_stride = chunk_size[1] * 960  # 600ms
    
    results = []
    cache = {}
    
    for chunk in chunks:
        chunk_audio = audio_np[chunk["start"]:chunk["end"]]
        total_chunk_num = int(len(chunk_audio - 1) / chunk_stride + 1)
        
        chunk_texts = []
        for i in range(total_chunk_num):
            speech_chunk = chunk_audio[i*chunk_stride:(i+1)*chunk_stride]
            is_final = i == total_chunk_num - 1
            
            try:
                res = stt_model.model.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=is_final,
                    chunk_size=chunk_size,
                    encoder_chunk_look_back=encoder_chunk_look_back,
                    decoder_chunk_look_back=decoder_chunk_look_back
                )
                
                if res and isinstance(res, list) and len(res) > 0:
                    if isinstance(res[0], dict) and "text" in res[0]:
                        text = rich_transcription_postprocess(res[0]["text"])
                        if text:
                            chunk_texts.append(text)
                            
            except Exception as e:
                print(click.style("ERROR", fg="red") + f":\t  Chunk transcription failed: {str(e)}")
                continue
                
        if chunk_texts:
            results.append("".join(chunk_texts))
            
    return " ".join(results)