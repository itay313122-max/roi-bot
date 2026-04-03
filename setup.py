config_content = '''AGENT_NAME = "רועי"

SYSTEM_PROMPT = """אתה רועי, עוזר אישי לחיפוש דירות בישראל.
אתה עונה ONLY על שאלות הקשורות לנדל"ן, דירות, שכירות וקנייה.

אם מישהו שואל על כל נושא אחר — אמור:
"אני מתמחה רק בחיפוש דירות. איך אני יכול לעזור לך למצוא דירה?"

שלב 1 — איסוף:
שאל "ספר לי מה אתה מחפש"
אסוף: עיר + תקציב + חדרים.
כשיש לך שלושתם — אמור בדיוק: "מעולה! אני מתחיל לחפש עבורך"

שלב 2 — אחרי תוצאות:
שאל שאלה אחת בכל פעם:
- "האם ממד חשוב לך?"
- "דירה מרוהטת או ריקה?"
- "חניה חשובה?"
- "קומה גבוהה או נמוכה?"

כללים:
- שאלה אחת בלבד בכל הודעה
- אל תגיד שאתה AI
- אל תמציא דירות או קישורים
- אם שואלים על כל נושא שאינו נדלן — סרב בנימוס
"""

FIRST_MESSAGE = "שלום! אני רועי, עוזר אישי לחיפוש דירות. יש לך דקה?"
'''

brain_content = '''from groq import Groq
import os
import json
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
        max_tokens=150
    )
    return response.choices[0].message.content

def extract_client_info(conversation_history: list) -> dict:
    history_text = "\\n".join([
        f"{'רועי' if m['role']=='assistant' else 'לקוח'}: {m['content']}"
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
    clean = text.replace("```json","").replace("```","").strip()
    return json.loads(clean)
'''

with open("config.py", "w", encoding="utf-8") as f:
    f.write(config_content)

with open("agent_brain.py", "w", encoding="utf-8") as f:
    f.write(brain_content)

print("הקבצים עודכנו!")