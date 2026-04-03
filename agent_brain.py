import json
from groq import Groq
import os
from dotenv import load_dotenv
from config import SYSTEM_PROMPT

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def think(conversation_history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=200
    )

    return response.choices[0].message.content


def extract_client_info(conversation_history: list) -> dict:
    history_text = "\n".join([
        f"{'רועי' if m['role'] == 'assistant' else 'לקוח'}: {m['content']}"
        for m in conversation_history
    ])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""חלץ מידע מהשיחה והחזר JSON בלבד:
{{
  "area": "אזור או עיר",
  "rooms": "מספר חדרים",
  "budget": "תקציב",
  "deal_type": "שכירה או קנייה",
  "parking": "כן/לא/לא ידוע",
  "furnished": "כן/לא/לא ידוע",
  "mamad": "כן/לא/לא ידוע",
  "floor": "גבוהה/נמוכה/לא ידוע",
  "notes": "כל מידע נוסף"
}}
אם מידע חסר כתוב לא ידוע.
השיחה:
{history_text}"""
        }]
    )
    text = response.choices[0].message.content
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)