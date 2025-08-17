#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрація клонування голосу з використанням аудіозапису з папки test
"""

import os
import sys
from pathlib import Path

# Імпорт з основного скрипта
from voice_cloning_script import VoiceCloner

def convert_ogg_to_wav(input_path: str, output_path: str) -> bool:
    """
    Конвертує OGG файл у WAV формат для клонування голосу
    
    Args:
        input_path: Шлях до OGG файлу
        output_path: Шлях для збереження WAV файлу
    
    Returns:
        bool: True якщо успішно, False якщо помилка
    """
    try:
        # Спроба використати librosa для конвертації
        import librosa
        import soundfile as sf
        
        print(f"🔄 Конвертація {input_path} → {output_path}")
        
        # Завантаження аудіо з librosa
        audio_data, sample_rate = librosa.load(input_path, sr=None)
        
        print(f"Параметри аудіо: sample_rate={sample_rate}, duration={len(audio_data)/sample_rate:.2f}s")
        
        # Збереження у WAV форматі
        sf.write(output_path, audio_data, sample_rate)
        
        # Перевірка результату
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Конвертація завершена: {output_path} ({file_size} байт)")
            return True
        else:
            print("❌ Файл не було створено після конвертації")
            return False
            
    except ImportError:
        print("❌ librosa не встановлено. Спроба альтернативного методу...")
        return convert_with_pydub(input_path, output_path)
    except Exception as e:
        print(f"❌ Помилка конвертації з librosa: {e}")
        return convert_with_pydub(input_path, output_path)

def convert_with_pydub(input_path: str, output_path: str) -> bool:
    """
    Альтернативна конвертація через pydub
    """
    try:
        from pydub import AudioSegment
        
        print("🔄 Конвертація через pydub...")
        
        # Завантаження OGG файлу
        audio = AudioSegment.from_ogg(input_path)
        
        # Конвертація у WAV
        audio.export(output_path, format="wav")
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Конвертація завершена: {output_path} ({file_size} байт)")
            return True
        else:
            return False
            
    except ImportError:
        print("❌ pydub не встановлено")
        print("💡 Встановіть: pip install pydub")
        return False
    except Exception as e:
        print(f"❌ Помилка конвертації з pydub: {e}")
        return False

def main():
    """
    Головна функція для демонстрації клонування голосу
    """
    print("=== Клонування голосу з аудіозапису ===\n")
    
    # Шляхи до файлів
    input_audio = "test/audio_2025-08-17_09-55-15.ogg"
    converted_audio = "test/speaker_sample.wav"
    
    # Перевірка існування вхідного файлу
    if not os.path.exists(input_audio):
        print(f"❌ Аудіофайл не знайдено: {input_audio}")
        return
    
    print(f"📁 Знайдено аудіофайл: {input_audio}")
    
    # 1. Конвертація OGG → WAV
    print("\n=== Етап 1: Конвертація аудіо ===")
    success = convert_ogg_to_wav(input_audio, converted_audio)
    
    if not success:
        print("❌ Не вдалося конвертувати аудіофайл")
        print("💡 Встановіть додаткові бібліотеки: pip install librosa soundfile pydub")
        return
    
    # 2. Ініціалізація VoiceCloner
    print("\n=== Етап 2: Ініціалізація клонера голосу ===")
    try:
        cloner = VoiceCloner()
        print("✅ VoiceCloner ініціалізовано")
    except Exception as e:
        print(f"❌ Помилка ініціалізації: {e}")
        print("💡 Встановіть Coqui TTS: pip install TTS")
        return
    
    # 3. Демонстрація клонування голосу
    print("\n=== Етап 3: Клонування голосу ===")
    
    # Тестові тексти для клонування
    test_texts = [
        {
            "text": "Привет! Это тест клонирования голоса на русском языке.",
            "language": "ru",  # XTTS не підтримує uk, використовуємо ru
            "output": "cloned_ukrainian.wav",
            "speed": 1.0
        },
        {
            "text": "Hello! This is a voice cloning test in English.",
            "language": "en", 
            "output": "cloned_english.wav",
            "speed": 1.0
        },
        {
            "text": "This is the same voice speaking faster than normal speed.",
            "language": "en",
            "output": "cloned_fast.wav", 
            "speed": 1.3
        },
        {
            "text": "This is the same voice speaking slowly and clearly.",
            "language": "en",
            "output": "cloned_slow.wav",
            "speed": 0.7
        }
    ]
    
    successful_clones = []
    failed_clones = []
    
    for i, test in enumerate(test_texts, 1):
        print(f"\n🎭 Тест {i}/4: {test['text'][:40]}...")
        print(f"   Мова: {test['language']}, Швидкість: {test['speed']}x")
        
        success = cloner.clone_voice_from_sample(
            text=test["text"],
            speaker_wav_path=converted_audio,
            output_path=test["output"],
            language=test["language"],
            speed=test["speed"]
        )
        
        if success:
            successful_clones.append(test["output"])
            print(f"   ✅ Збережено: {test['output']}")
        else:
            failed_clones.append(test["output"])
            print(f"   ❌ Помилка: {test['output']}")
    
    # 4. Результати
    print("\n=== Результати клонування ===")
    print(f"✅ Успішно: {len(successful_clones)}")
    print(f"❌ Помилок: {len(failed_clones)}")
    
    if successful_clones:
        print("\nЗгенеровані файли:")
        for file in successful_clones:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"  - {file} ({size} байт)")
    
    if failed_clones:
        print("\nПомилки:")
        for file in failed_clones:
            print(f"  - {file}")
    
    print(f"\n📊 Оригінальний зразок: {converted_audio}")
    print("🎧 Порівняйте оригінал з клонованими версіями для оцінки якості")
    
    # 5. Додаткова інформація
    print("\n=== Рекомендації ===")
    print("🔍 Для кращої якості клонування:")
    print("  - Використовуйте чистий запис без шумів")
    print("  - Оптимальна тривалість: 6-15 секунд")
    print("  - Голос має бути виразним та чітким")
    print("  - Уникайте музики або фонових звуків")

if __name__ == "__main__":
    main()
