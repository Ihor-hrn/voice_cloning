# Клонування голосу з Coqui XTTS v2

Демонстрація клонування голосу з використанням відкритих AI-моделей.

## Встановлення

```bash
pip install TTS pandas numpy torch torchaudio
```

## Використання

```python
from voice_cloning_script import VoiceCloner

# Ініціалізація з XTTS v2 для клонування
cloner = VoiceCloner()

# Простий TTS
cloner.simple_text_to_speech("Hello world!", "output.wav")

# Клонування голосу
cloner.clone_voice_from_sample(
    text="This is cloned voice speaking",
    speaker_wav_path="speaker_sample.wav",
    output_path="cloned.wav",
    language="en",
    speed=1.0
)
```

## Демонстрація

Запустіть `python voice_cloning_script.py` для повної демонстрації:

✅ **Простий TTS** - генерація з одного речення  
✅ **Деталізований TTS** - обробка CSV/JSON файлів  
✅ **Клонування голосу** - з різними швидкостями та мовами  
✅ **Крос-мовне клонування** - голос однією мовою, текст іншою  

## Підтримувані мови XTTS v2

en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi

## Вимоги до зразка голосу

- Формат: WAV
- Тривалість: 6-15 секунд
- Якість: чисте мовлення без шуму