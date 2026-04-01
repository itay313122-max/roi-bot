import base64
import os
import httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_image_from_url(image_url: str) -> dict:
    image_data = base64.standard_b64encode(
        httpx.get(image_url).content
    ).decode("utf-8")

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": """Analyze this real estate photo. Reply in Hebrew with JSON only:
{
  "room_type": "type of room",
  "condition": "new/good/fair/needs_renovation",
  "size": "large/medium/small",
  "light": "bright/medium/dark",
  "issues": ["list of visible problems"],
  "positives": ["list of good things"],
  "buyer_note": "one sentence advice for buyer in Hebrew"
}"""
                }
            ]
        }],
        max_tokens=400
    )

    import json
    text = response.choices[0].message.content
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def analyze_multiple_images(urls: list) -> str:
    results = []
    for url in urls:
        try:
            result = analyze_image_from_url(url)
            results.append(result)
            print(f"תמונה נותחה: {result.get('room_type', 'לא ידוע')}")
        except Exception as e:
            print(f"שגיאה בתמונה: {e}")

    issues = []
    positives = []
    for r in results:
        issues.extend(r.get("issues", []))
        positives.extend(r.get("positives", []))

    summary = f"""
סיכום ניתוח {len(results)} תמונות:
יתרונות: {', '.join(positives[:5]) if positives else 'לא זוהו'}
בעיות: {', '.join(issues[:5]) if issues else 'לא זוהו'}
המלצה: {'שווה ביקור' if len(issues) < 3 else 'יש לבדוק לעומק'}
"""
    return summary