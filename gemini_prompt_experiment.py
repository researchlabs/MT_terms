import pandas as pd
import time
import os
import re
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SOURCE_LANG = "English"
TARGET_LANG = "Ukrainian"

PROMPT_VARIANTS = {
    "base": f"You are a professional medical translator. Translate from {SOURCE_LANG} to {TARGET_LANG}. Return ONLY the translation, nothing else.",
    "formal": f"You are a professional medical translator. Translate from {SOURCE_LANG} to {TARGET_LANG} using precise and formal medical terminology. Prefer full technical terms over abbreviations. Return ONLY the translation, nothing else.",
    "patient": f"You are a professional medical translator. Translate from {SOURCE_LANG} to {TARGET_LANG} in a way that is easy for a patient to understand. Use simple terms. Return ONLY the translation, nothing else.",
    "knowledge": f"You are a professional medical translator. Use the provided medical knowledge to ensure accurate and consistent translation. Translate from {SOURCE_LANG} to {TARGET_LANG}. Return ONLY the translation, nothing else."
}

KNOWLEDGE_MAP = {
    "ASA": "ASA = acetylsalicylic acid (aspirin)\nGI bleeding = gastrointestinal bleeding",
    "gr": "gr = grain (64.8 mg)\nTID = three times a day",
    "tsp": "tsp = 5 ml\nPO = orally\nTID = three times a day",
    "MMR": "MMR = measles, mumps, rubella\nmeasles = кір\nmumps = паротит\nrubella = краснуха",
    "GLP1": "GLP-1 = glucagon-like peptide-1"
}

def get_knowledge(text):
    if "ASA" in text:
        return KNOWLEDGE_MAP["ASA"]

    elif re.search(r"\bgr\b", text):
        return KNOWLEDGE_MAP["gr"]

    elif "tsp" in text:
        return KNOWLEDGE_MAP["tsp"]

    elif "MMR" in text:
        return KNOWLEDGE_MAP["MMR"]

    elif "GLP-1" in text:
        return KNOWLEDGE_MAP["GLP1"]

    return ""


def translate_gemini(text, prompt_type):
    try:
        system_prompt = PROMPT_VARIANTS[prompt_type]

        if prompt_type == "knowledge":
            knowledge = get_knowledge(text)
            full_prompt = f"{system_prompt}\n\nMedical knowledge:\n{knowledge}\n\nText: {text}"
        else:
            full_prompt = f"{system_prompt}\n\nText: {text}"

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=full_prompt
        )

        return response.text.strip()

    except Exception as e:
        return f"Error: {e}"


def main():
    file_path = "./data/texts_prompt.csv"
    output_file = "./data/gemini_prompt_experiment.csv"
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

    results = {
        "base": [],
        "formal": [],
        "patient": [],
        "knowledge": [],
    }

    for index, row in test_df.iterrows():
        text = str(row[column_name])
        print(f"Обробка рядка {index + 1}: {text[:60]}...")

        for key in results.keys():
            retry_delay = 5
            max_retries = 5
            retry_count = 0
            success = False

            while not success and retry_count < max_retries:
                translation = translate_gemini(text, key)
                if "RESOURCE_EXHAUSTED" in translation or "UNAVAILABLE" in translation:
                    retry_count += 1
                    print(f"  Квота або модель недоступна для '{key}', спроба {retry_count}/{max_retries}, повтор через {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)
                else:
                    success = True
                    results[key].append(translation)
                    print(f"  Переклад для '{key}' завершено")
                    time.sleep(3)

            if not success:
                results[key].append(f"Error: Quota still exceeded after {max_retries} retries")
                print(f"  Переклад для '{key}' не вдалося після {max_retries} спроб")

    for key, values in results.items():
        test_df[f"Gemini_{key}"] = values

    test_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\nРезультати збережено у файлі: {output_file}")


if __name__ == "__main__":
    main()