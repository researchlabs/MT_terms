import pandas as pd
import time
import os
import re
import json
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SOURCE_LANG = "English"
TARGET_LANG = "Ukrainian"

def load_knowledge_graph(file_path):
    """Завантажує граф знань із JSON-файлу."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("knowledge_base", [])
    except Exception as e:
        print(f"Помилка завантаження графа: {e}")
        return []

def get_relevant_context(text, kb):
    """
    Шукає в тексті терміни, що є в графі знань.
    Повертає лише ті вузли, які стосуються цього конкретного речення.
    """
    relevant_nodes = []
    for entry in kb:
        pattern = rf"\b{re.escape(entry['term'])}\b"
        if re.search(pattern, text, re.IGNORECASE):
            relevant_nodes.append(entry)
    return relevant_nodes


def translate_with_knowledge(text, kb):   
    context = get_relevant_context(text, kb)
    context_str = json.dumps(context, ensure_ascii=False) if context else "No specific medical entities found"

    system_prompt = (
        f"You are a professional medical translator. "
        f"Use the following STRUCTURED KNOWLEDGE extracted from a Knowledge Graph to ensure accuracy:\n"
        f"KNOWLEDGE CONTEXT: {context_str}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Translate accurately from {SOURCE_LANG} to {TARGET_LANG}\n"
        f"- Prioritize terminology and dosages from the KNOWLEDGE CONTEXT\n"
        f"- Return ONLY the translated text"
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"{system_prompt}\n\nText to translate: {text}"
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"


def main():
    kb = load_knowledge_graph("./data/knowledge_graph.json")
    
    input_file = "./data/texts_prompt.csv"
    # output_file = "./data/knowledge_graph_results.csv"
    output_file = "./data/gemini_prompt_experiment.csv"
    
    if not os.path.exists(input_file):
        print(f"Помилка: Вхідний файл {input_file} не знайдено!")
        return

    df_input = pd.read_csv(input_file)

    if os.path.exists(output_file):
        df_result = pd.read_csv(output_file)
    else:
        df_result = df_input.copy()

    translations = []
    column_name = 'Gemini_graph'

    print(f"Початок перекладу {len(df_input)} текстів...")

    for index, row in df_input.iterrows():
        text = str(row['text'])
        print(f"Обробка рядка {index + 1}/{len(df_input)}...")
        
        result = translate_with_knowledge(text, kb)
        translations.append(result)
        
        time.sleep(15) 

    df_result[column_name] = translations
    
    df_result.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nГотово! Результати збережено в {output_file}")


if __name__ == "__main__":
    main()