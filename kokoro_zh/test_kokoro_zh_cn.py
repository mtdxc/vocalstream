from kokoro_zh_cn import get_tts_model, KokoroTTSOptions
import soundfile as sf

def test_tts():
    # 初始化 TTS 模型
    tts_model = get_tts_model()
    
    # 测试文本
    texts = [
        "千里之行，始于足下。",
        "学而不思则罔，思而不学则殆。",
        "人工智能正在改变我们的生活方式。",
    ]
    
    # 测试不同的语音和速度
    test_cases = [
        {"voice": "zf_001", "speed": 1.0},
        {"voice": "zf_001", "speed": 1.2},
        {"voice": "zf_002", "speed": 1.0},
    ]
    
    for i, text in enumerate(texts):
        for j, case in enumerate(test_cases):
            # 设置 TTS 选项
            options = KokoroTTSOptions(
                voice=case["voice"],
                speed=case["speed"]
            )
            
            print(f"\n测试用例 {i+1}-{j+1}:")
            print(f"文本: {text}")
            print(f"语音: {case['voice']}")
            print(f"速度: {case['speed']}")
            
            # 生成语音
            sample_rate, audio = tts_model.tts(text, options)
            
            # 保存音频文件
            filename = f"test_output_{i+1}_{j+1}.wav"
            sf.write(filename, audio, sample_rate)
            print(f"已生成音频文件: {filename}")

if __name__ == "__main__":
    test_tts()