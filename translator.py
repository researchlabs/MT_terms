import pandas as pd
import time
import os
from dotenv import load_dotenv
import google.genai as genai
from openai import OpenAI, RateLimitError

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

SOURCE_LANG = "English"
TARGET_LANG = "Ukrainian"
SYSTEM_PROMPT = f"You are a professional medical translator. Translate from {SOURCE_LANG} to {TARGET_LANG}. Return ONLY the translation, nothing else."

def translate_gemini(text):
    try:
        prompt = f"{SYSTEM_PROMPT}\n\nText: {text}"
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def translate_openai(text):
    try:
        response = openai_client.responses.create(
            model="gpt-5-nano",
            input=f"{SYSTEM_PROMPT}\n\nText: {text}"
        )
        return response.output_text.strip()
    except RateLimitError:
        return "OpenAI Error: Rate limit/Quota exceeded"
    except Exception as e:
        return f"OpenAI Error: {e}"


def main():
    file_path = "./data/texts.csv"
    output_file = "./data/comparison_results.csv"
    column_name = "text" 

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Не вдалося відкрити файл: {e}")
        return

    if column_name not in df.columns:
        print(f"'{column_name}' не знайдено! Доступні: {list(df.columns)}")
        return

    test_df = df.copy()
    
    gemini_translations = []
    openai_translations = []

    for index, row in test_df.iterrows():
        text = str(row[column_name])
        print(f"Обробка рядка {index + 1}...")

        gemini_translations.append(translate_gemini(text))
        openai_translations.append(translate_openai(text))

        time.sleep(4)

    test_df['Gemini_Translation'] = gemini_translations
    test_df['GPT5_Translation'] = openai_translations

    test_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nРезультати збережено у файлі: {output_file}")

if __name__ == "__main__":
    main()