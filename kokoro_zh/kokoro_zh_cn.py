import asyncio
import re
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol, TypeVar

import numpy as np
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from misaki import zh
from kokoro_onnx import Kokoro

class TTSOptions:
    pass

T = TypeVar("T", bound=TTSOptions, contravariant=True)


class TTSModel(Protocol[T]):
    def tts(
        self, text: str, options: T | None = None
    ) -> tuple[int, NDArray[np.float32]]: ...

    def stream_tts(
        self, text: str, options: T | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]: ...

    def stream_tts_sync(
        self, text: str, options: T | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]: ...


@dataclass
class KokoroTTSOptions(TTSOptions):
    # voice: str = "af_heart"
    # speed: float = 1.0
    # lang: str = "en-us"
    voice: str = "zf_001"
    speed: float = 1.0
    # lang: str = "en-us"


@lru_cache
def get_tts_model(model: Literal["kokoro"] = "kokoro") -> TTSModel:
    m = KokoroTTSModel()
    m.tts("千里之行，始于足下。")
    # m.tts("Hello, world!")
    return m


class KokoroFixedBatchSize:
    # Source: https://github.com/thewh1teagle/kokoro-onnx/issues/115#issuecomment-2676625392
    def _split_phonemes(self, phonemes: str) -> list[str]:
        MAX_PHONEME_LENGTH = 510
        max_length = MAX_PHONEME_LENGTH - 1
        batched_phonemes = []
        while len(phonemes) > max_length:
            # Find best split point within limit
            split_idx = max_length

            # Try to find the last period before max_length
            period_idx = phonemes.rfind(".", 0, max_length)
            if period_idx != -1:
                split_idx = period_idx + 1  # Include period

            else:
                # Try other punctuation
                match = re.search(
                    r"[!?;,]", phonemes[:max_length][::-1]
                )  # Search backwards
                if match:
                    split_idx = max_length - match.start()

                else:
                    # Try last space
                    space_idx = phonemes.rfind(" ", 0, max_length)
                    if space_idx != -1:
                        split_idx = space_idx

            # If no good split point is found, force split at max_length
            chunk = phonemes[:split_idx].strip()
            batched_phonemes.append(chunk)

            # Move to the next part
            phonemes = phonemes[split_idx:].strip()

        # Add remaining phonemes
        if phonemes:
            batched_phonemes.append(phonemes)
        return batched_phonemes


class KokoroTTSModel(TTSModel):
    def __init__(self):

        self.model = Kokoro(
            # model_path=hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx"),
            # voices_path=hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin"),
            # 设置自定义模型位置
            model_path="E:\\Code\\PythonDir\\TTS\\VocalStream\\Model\\kokoro-v1.1-zh.onnx",
            voices_path="E:\\Code\\PythonDir\\TTS\\VocalStream\\Model\\voices-v1.1-zh.bin",
            vocab_config="E:\\Code\\PythonDir\\TTS\\VocalStream\\Model\\config.json"
        )
        # 仅针对中文
        self.g2p = zh.ZHG2P(version="1.1")
        self.model._split_phonemes = KokoroFixedBatchSize()._split_phonemes

    def tts(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32]]:
        options = options or KokoroTTSOptions()
        txt, _ = self.g2p(text)
        # 结果与速率
        a, b = self.model.create(
            txt, voice=options.voice, speed=options.speed, is_phonemes=True # lang=options.lang,
        )
        return b, a

    async def stream_tts(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        options = options or KokoroTTSOptions()

        # 修改分句正则表达式，增加中文标点支持
        sentences = re.split(r"(?<=[.!?。！？])\s*", text.strip())

        for s_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            chunk_idx = 0
            # print(sentence)
            # 中文支持
            txt, _ = self.g2p(sentence)
            print(txt)
            async for chunk in self.model.create_stream(
                txt, voice=options.voice, speed=options.speed, is_phonemes=True # lang=options.lang
            ):
                if s_idx != 0 and chunk_idx == 0:
                    yield chunk[1], np.zeros(chunk[1] // 7, dtype=np.float32)
                chunk_idx += 1
                yield chunk[1], chunk[0]

    def stream_tts_sync(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        loop = asyncio.new_event_loop()

        # Use the new loop to run the async generator
        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break