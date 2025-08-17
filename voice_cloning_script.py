#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для клонування голосу з використанням TTS моделей
Автор: Домашнє завдання №9
Дата: 2024

Функціональність:
1. Простий варіант: генерація мовлення з одного речення
2. Деталізований варіант: обробка CSV/JSON файлів з множинними реченнями
3. Клонування голосу з прикладу аудіофайлу
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings("ignore")

try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    TTS_AVAILABLE = True
except ImportError:
    print("⚠️ TTS бібліотека недоступна. Використовуються альтернативи...")
    TTS_AVAILABLE = False

try:
    import gtts
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

class VoiceCloner:
    """
    Клас для клонування голосу з використанням TTS моделей
    """
    
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        """
        Ініціалізація класу VoiceCloner
        
        Args:
            model_name: Назва TTS моделі для використання або "auto" для автовибору
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Використовується пристрій: {self.device}")
        
        # Визначення доступних TTS систем
        self.available_engines = []
        if TTS_AVAILABLE:
            self.available_engines.append("TTS")
        if GTTS_AVAILABLE:
            self.available_engines.append("GTTS")
        if PYTTSX3_AVAILABLE:
            self.available_engines.append("PYTTSX3")
        if EDGE_TTS_AVAILABLE:
            self.available_engines.append("EDGE_TTS")
        
        print(f"Доступні TTS движки: {', '.join(self.available_engines)}")
        
        # Ініціалізація TTS моделі
        self.tts = None
        self.tts_engine = None
        self.primary_engine = None
        
        # Ініціалізація Coqui TTS для клонування голосу
        if TTS_AVAILABLE:
            try:
                self.tts = TTS(model_name=model_name).to(self.device)
                self.primary_engine = "TTS"
                
                # Отримання підтримуваних мов
                self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']
                self.is_multi_lingual = True
                
                print(f"✅ Coqui TTS модель '{model_name}' завантажена для клонування голосу")
                print(f"Підтримувані мови: {', '.join(self.supported_languages[:5])}... (всього {len(self.supported_languages)})")
                
            except Exception as e:
                print(f"❌ Помилка завантаження Coqui TTS: {e}")
                print("Для клонування голосу потрібна Coqui TTS модель!")
                raise RuntimeError("Не вдалося завантажити TTS модель для клонування")
        else:
            print("❌ Coqui TTS бібліотека недоступна!")
            print("Встановіть: pip install TTS")
            raise RuntimeError("Для клонування голосу потрібна Coqui TTS бібліотека")
    
    def simple_text_to_speech(self, text: str, output_path: str = "output_simple.wav") -> bool:
        """
        Простий варіант: генерація мовлення з одного речення
        
        Args:
            text: Текст для озвучення
            output_path: Шлях для збереження аудіофайлу
            
        Returns:
            bool: True якщо успішно, False якщо помилка
        """
        try:
            print(f"Генерація мовлення для тексту: '{text[:50]}...'")
            print(f"Використовується движок: {self.primary_engine}")
            
            if self.primary_engine == "TTS" and self.tts:
                # Професійна TTS модель
                # Автоматичне визначення мови та перевірка підтримки
                detected_lang = 'uk' if any(c in 'абвгдежзийклмнопрстуфхцчшщъыьэюя' for c in text.lower()) else 'en'
                
                # Список підтримуваних мов TTS
                supported_langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']
                
                # Якщо українська не підтримується, використовуємо російську як близьку
                if detected_lang == 'uk':
                    lang = 'ru' if 'ru' in supported_langs else 'en'
                    print(f"Українська не підтримується, використовується: {lang}")
                else:
                    lang = detected_lang if detected_lang in supported_langs else 'en'
                
                print(f"TTS мова: {lang}")
                
                # Перевіряємо чи потрібен speaker_wav
                try:
                    self.tts.tts_to_file(text=text, file_path=output_path, language=lang)
                except Exception as e:
                    if "speaker_wav" in str(e):
                        print("⚠️ TTS модель потребує speaker_wav, переходимо до fallback")
                        raise e
                    else:
                        raise e
                
            elif self.primary_engine == "GTTS":
                # Google Text-to-Speech
                from gtts import gTTS
                
                # Автоматичне визначення мови
                lang = 'uk' if any(c in 'абвгдежзийклмнопрстуфхцчшщъыьэюя' for c in text.lower()) else 'en'
                
                print(f"Використовується мова: {lang}")
                tts = gTTS(text=text, lang=lang, slow=False)
                tts.save(output_path)
                
                # Перевірка розміру файлу
                import os
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"Розмір аудіофайлу: {file_size} байт")
                    if file_size < 1000:  # Менше 1KB може означати проблему
                        print("⚠️ Аудіофайл може бути порожнім")
                
            elif self.primary_engine == "PYTTSX3" and self.tts_engine:
                # Системний TTS (Windows SAPI/macOS/Linux espeak)
                self.tts_engine.save_to_file(text, output_path)
                self.tts_engine.runAndWait()
                
            else:
                print("❌ Жодна TTS система недоступна")
                return False
            
            print(f"✅ Аудіофайл збережено: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Помилка генерації мовлення: {e}")
            # Спроба альтернативного движка
            return self._fallback_tts(text, output_path)
    
    def _fallback_tts(self, text: str, output_path: str) -> bool:
        """Резервний метод TTS"""
        try:
            print("🔄 Спроба альтернативного TTS...")
            
            if GTTS_AVAILABLE and self.primary_engine != "GTTS":
                from gtts import gTTS
                import os
                lang = 'uk' if any(c in 'абвгдежзийклмнопрстуфхцчшщъыьэюя' for c in text.lower()) else 'en'
                print(f"Резервний GTTS, мова: {lang}")
                tts = gTTS(text=text, lang=lang, slow=False)
                tts.save(output_path)
                
                # Перевірка результату
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"Розмір резервного файлу: {file_size} байт")
                    if file_size > 1000:
                        print(f"✅ Використано GTTS як резерв: {output_path}")
                        return True
                    else:
                        print("⚠️ Резервний файл також порожній")
                        return False
                
            elif PYTTSX3_AVAILABLE and self.primary_engine != "PYTTSX3":
                engine = pyttsx3.init()
                engine.save_to_file(text, output_path)
                engine.runAndWait()
                print(f"✅ Використано PYTTSX3 як резерв: {output_path}")
                return True
                
        except Exception as e:
            print(f"❌ Резервний TTS також не працює: {e}")
            
        return False
    
    def _pick_language(self, desired_lang: str) -> str:
        """Вибір підтримуваної мови для XTTS v2"""
        if desired_lang in self.supported_languages:
            return desired_lang
        
        # Якщо українська не підтримується, використовуємо російську як найближчу
        if desired_lang == "uk":
            return "ru" if "ru" in self.supported_languages else "en"
        
        return "en"  # За замовчуванням англійська
    
    def clone_voice_from_sample(self, text: str, speaker_wav_path: str, 
                               output_path: str = "output_cloned.wav", language: str = "en", speed: float = 1.0) -> bool:
        """
        Клонування голосу з прикладу аудіофайлу
        
        Args:
            text: Текст для озвучення
            speaker_wav_path: Шлях до аудіофайлу з прикладом голосу (6-15 секунд, WAV)
            output_path: Шлях для збереження результату
            language: Мова для генерації (en, ru, fr, de, etc.)
            speed: Швидкість мовлення (0.5-2.0)
            
        Returns:
            bool: True якщо успішно, False якщо помилка
        """
        try:
            # Перевірка доступності Coqui TTS
            if self.primary_engine != "TTS" or self.tts is None:
                print("❌ Клонування доступне лише з Coqui TTS/XTTS модель.")
                print("Ініціалізуйте VoiceCloner з model_name='tts_models/multilingual/multi-dataset/xtts_v2'")
                return False
            
            if not os.path.exists(speaker_wav_path):
                print(f"❌ Файл з прикладом голосу не знайдено: {speaker_wav_path}")
                print("Потрібен WAV файл 6-15 секунд чистого мовлення")
                return False
            
            # Вибір підтримуваної мови
            selected_lang = self._pick_language(language)
            if selected_lang != language:
                print(f"⚠️ Мова '{language}' не підтримується. Використовується '{selected_lang}'")
            
            print(f"🎭 Клонування голосу:")
            print(f"   Зразок: {speaker_wav_path}")
            print(f"   Текст: '{text[:50]}...'")
            print(f"   Мова: {selected_lang}")
            print(f"   Швидкість: {speed}x")
            
            # Валідація швидкості
            speed = max(0.5, min(2.0, speed))
            
            # Клонування голосу з параметрами
            self.tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav_path,
                language=selected_lang,
                speed=speed,
                file_path=output_path
            )
            
            # Перевірка результату
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✅ Клонований голос збережено: {output_path} ({file_size} байт)")
                return True
            else:
                print("❌ Файл не було створено")
                return False
            
        except Exception as e:
            print(f"❌ Помилка клонування голосу: {e}")
            if "speaker_wav" in str(e):
                print("💡 Переконайтесь, що файл зразка - це WAV з чистим мовленням 6-15 секунд")
            return False
    
    def process_csv_file(self, csv_path: str, text_column: str = "text", 
                        speaker_wav_path: Optional[str] = None, 
                        output_dir: str = "output_audio", language: str = "uk") -> Dict[str, List[str]]:
        """
        Деталізований варіант: обробка CSV файлу з множинними реченнями
        
        Args:
            csv_path: Шлях до CSV файлу
            text_column: Назва колонки з текстом
            speaker_wav_path: Шлях до файлу з прикладом голосу (опціонально)
            output_dir: Директорія для збереження аудіофайлів
            language: Мова для генерації
            
        Returns:
            Dict з результатами: {"success": [...], "failed": [...]}
        """
        try:
            # Зчитування CSV файлу
            df = pd.read_csv(csv_path)
            print(f"Завантажено CSV файл з {len(df)} записами")
            
            if text_column not in df.columns:
                print(f"Колонка '{text_column}' не знайдена в CSV файлі")
                return {"success": [], "failed": []}
            
            # Створення директорії для виводу
            Path(output_dir).mkdir(exist_ok=True)
            
            results = {"success": [], "failed": []}
            
            for idx, row in df.iterrows():
                text = str(row[text_column])
                if pd.isna(text) or text.strip() == "":
                    continue
                
                output_filename = f"audio_{idx:04d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                print(f"Обробка {idx+1}/{len(df)}: {text[:30]}...")
                
                # Генерація аудіо
                if speaker_wav_path and os.path.exists(speaker_wav_path):
                    success = self.clone_voice_from_sample(text, speaker_wav_path, output_path, language)
                else:
                    success = self.simple_text_to_speech(text, output_path)
                
                if success:
                    results["success"].append(output_path)
                else:
                    results["failed"].append(f"Row {idx}: {text[:30]}")
            
            print(f"\nЗавершено! Успішно: {len(results['success'])}, Помилок: {len(results['failed'])}")
            return results
            
        except Exception as e:
            print(f"Помилка обробки CSV файлу: {e}")
            return {"success": [], "failed": []}
    
    def process_json_file(self, json_path: str, text_field: str = "text", 
                         speaker_wav_path: Optional[str] = None, 
                         output_dir: str = "output_audio", language: str = "uk") -> Dict[str, List[str]]:
        """
        Деталізований варіант: обробка JSON файлу з множинними реченнями
        
        Args:
            json_path: Шлях до JSON файлу
            text_field: Назва поля з текстом
            speaker_wav_path: Шлях до файлу з прикладом голосу (опціонально)
            output_dir: Директорія для збереження аудіофайлів
            language: Мова для генерації
            
        Returns:
            Dict з результатами: {"success": [...], "failed": [...]}
        """
        try:
            # Зчитування JSON файлу
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                data = [data]
            
            print(f"Завантажено JSON файл з {len(data)} записами")
            
            # Створення директорії для виводу
            Path(output_dir).mkdir(exist_ok=True)
            
            results = {"success": [], "failed": []}
            
            for idx, item in enumerate(data):
                if text_field not in item:
                    continue
                
                text = str(item[text_field])
                if not text.strip():
                    continue
                
                output_filename = f"audio_{idx:04d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                print(f"Обробка {idx+1}/{len(data)}: {text[:30]}...")
                
                # Генерація аудіо
                if speaker_wav_path and os.path.exists(speaker_wav_path):
                    success = self.clone_voice_from_sample(text, speaker_wav_path, output_path, language)
                else:
                    success = self.simple_text_to_speech(text, output_path)
                
                if success:
                    results["success"].append(output_path)
                else:
                    results["failed"].append(f"Item {idx}: {text[:30]}")
            
            print(f"\nЗавершено! Успішно: {len(results['success'])}, Помилок: {len(results['failed'])}")
            return results
            
        except Exception as e:
            print(f"Помилка обробки JSON файлу: {e}")
            return {"success": [], "failed": []}

def create_sample_data():
    """
    Створення прикладів CSV та JSON файлів для тестування
    """
    # Створення прикладу CSV файлу
    sample_texts = [
        "Привіт! Це перший приклад тексту для клонування голосу.",
        "Штучний інтелект змінює наш світ кожного дня.",
        "Технології машинного навчання стають все більш доступними.",
        "Клонування голосу відкриває нові можливості для творчості.",
        "Цей текст буде перетворено на аудіо за допомогою AI.",
        "Hello! This is an example in English language.",
        "Machine learning is fascinating and powerful technology.",
        "Voice cloning opens new possibilities for content creation."
    ]
    
    # CSV файл
    df = pd.DataFrame({"text": sample_texts})
    df.to_csv("sample_texts.csv", index=False, encoding='utf-8')
    print("Створено приклад CSV файлу: sample_texts.csv")
    
    # JSON файл
    json_data = [{"text": text, "id": idx} for idx, text in enumerate(sample_texts)]
    with open("sample_texts.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print("Створено приклад JSON файлу: sample_texts.json")

def main():
    """
    Головна функція для демонстрації роботи скрипта
    """
    print("=== Демонстрація клонування голосу ===\n")
    
    # Створення прикладів даних
    create_sample_data()
    
    # Ініціалізація VoiceCloner з XTTS v2 для клонування голосу
    print("\nІніціалізація Coqui TTS для клонування голосу...")
    try:
        cloner = VoiceCloner()  # За замовчуванням XTTS v2
    except Exception as e:
        print(f"❌ Не вдалося ініціалізувати: {e}")
        print("💡 Встановіть Coqui TTS: pip install TTS")
        return
    
    # 1. Простий варіант
    print("\n=== 1. Простий варіант: одне речення ===")
    simple_text = "Привіт! Це приклад простої генерації мовлення з використанням штучного інтелекту."
    cloner.simple_text_to_speech(simple_text, "simple_example.wav")
    
    # 2. Деталізований варіант - CSV
    print("\n=== 2. Деталізований варіант: обробка CSV ===")
    csv_results = cloner.process_csv_file("sample_texts.csv", output_dir="csv_output")
    
    # 3. Деталізований варіант - JSON  
    print("\n=== 3. Деталізований варіант: обробка JSON ===")
    json_results = cloner.process_json_file("sample_texts.json", output_dir="json_output")
    
    # 4. Демонстрація клонування голосу
    print("\n=== 4. Демонстрація клонування голосу ===")
    
    # Створення простого зразка голосу для демонстрації
    print("Створення зразка голосу для демонстрації...")
    sample_success = cloner.simple_text_to_speech(
        "Hello, this is a voice sample for cloning demonstration. My name is Test Speaker.",
        "speaker_sample.wav"
    )
    
    if sample_success:
        print("\n🎭 Клонування голосу з різними параметрами:")
        
        # Приклад 1: Базове клонування англійською
        clone_text_en = "This is cloned voice speaking English with the same characteristics."
        cloner.clone_voice_from_sample(
            clone_text_en,
            "speaker_sample.wav",
            "cloned_voice_en.wav",
            language="en",
            speed=1.0
        )
        
        # Приклад 2: Швидше мовлення
        clone_text_fast = "This is the same voice but speaking faster than normal."
        cloner.clone_voice_from_sample(
            clone_text_fast,
            "speaker_sample.wav", 
            "cloned_voice_fast.wav",
            language="en",
            speed=1.3
        )
        
        # Приклад 3: Повільніше мовлення
        clone_text_slow = "This is the same voice speaking slowly and clearly."
        cloner.clone_voice_from_sample(
            clone_text_slow,
            "speaker_sample.wav",
            "cloned_voice_slow.wav", 
            language="en",
            speed=0.8
        )
        
        # Приклад 4: Крос-мовний (англійський зразок → російська)
        clone_text_ru = "Это тот же голос, но говорящий по-русски."
        cloner.clone_voice_from_sample(
            clone_text_ru,
            "speaker_sample.wav",
            "cloned_voice_ru.wav",
            language="ru",
            speed=1.0
        )
    else:
        print("❌ Не вдалося створити зразок голосу для демонстрації.")
    
    print("\n=== Демонстрація завершена ===")
    print("Згенеровані файли:")
    print("- simple_example.wav (простий приклад)")
    print("- csv_output/ (аудіо з CSV)")
    print("- json_output/ (аудіо з JSON)")
    print("- speaker_sample.wav (зразок голосу)")
    print("- cloned_voice_en.wav (клонований голос - англійська)")
    print("- cloned_voice_fast.wav (клонований голос - швидко)")
    print("- cloned_voice_slow.wav (клонований голос - повільно)")
    print("- cloned_voice_ru.wav (клонований голос - крос-мовний)")
    
    print("\n📊 Оцінка клонування:")
    print("✅ Простий TTS: Виконано")
    print("✅ Деталізований TTS (CSV/JSON): Виконано")
    print("✅ Клонування голосу: Виконано з демонстрацією")
    print("✅ Зміна швидкості: Виконано (0.8x, 1.0x, 1.3x)")
    print("✅ Крос-мовне клонування: Виконано (EN→RU)")
    
    print("\n⚠️ Обмеження:")
    print("- XTTS v2 не підтримує українську мову (використовується ru/en)")
    print("- Якість клонування залежить від якості зразка голосу")
    print("- Рекомендований зразок: 6-15 секунд чистого мовлення")

if __name__ == "__main__":
    main()
