import requests
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

CITY_MAP = {
    "תל אביב": "tel-aviv",
    "ירושלים": "jerusalem",
    "חיפה": "haifa",
    "רמת גן": "ramat-gan",
    "פתח תקווה": "petah-tikva",
    "ראשון לציון": "rishon-lezion",
    "נתניה": "netanya",
    "באר שבע": "beer-sheva",
    "הרצליה": "herzliya",
    "רעננה": "raanana",
    "גבעתיים": "givatayim",
    "חולון": "holon",
    "בת ים": "bat-yam"
}

def parse_price(text: str) -> int:
    numbers = re.findall(r'\d+', text.replace(",", ""))
    for n in numbers:
        try:
            val = int(n)
            if 1000 < val < 100000:
                return val
        except:
            continue
    return 999999

def parse_rooms(text: str) -> float:
    match = re.search(r'(\d+\.?\d*)\s*room', text.lower())
    if match:
        return float(match.group(1))
    return 0

def scrape_rss(city: str, rooms: str, budget: int) -> list:
    url = "https://realta.co.il/feed.xml"
    print(f"סורק RSS של Realta...")

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        root = ET.fromstring(response.content)

        try:
            rooms_num = float(str(rooms).replace("חדרים", "").replace("חדר", "").strip())
        except:
            rooms_num = 3

        city_slug = CITY_MAP.get(city, city.lower().replace(" ", "-"))

        properties = []
        items = root.findall(".//item")
        print(f"סה״כ פריטים ב-RSS: {len(items)}")

        for item in items:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            desc = item.findtext("description") or ""
            image_el = item.find("enclosure")
            image = image_el.get("url") if image_el is not None else ""

            if city_slug not in link.lower():
                continue

            price = parse_price(title)

            if price > budget:
                continue

            prop_rooms = parse_rooms(title)
            if prop_rooms > 0 and abs(prop_rooms - rooms_num) > 0.5:
                continue

            properties.append({
                "title": title,
                "url": link,
                "description": desc,
                "price": price,
                "image": image
            })

        properties.sort(key=lambda x: x["price"])
        print(f"נמצאו {len(properties)} דירות מתאימות")
        return properties

    except Exception as e:
        print(f"שגיאה בסריקה: {e}")
        return []

def deep_analyze_property(url: str, title: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)[:3000]

        analysis = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"""נתח את דף הדירה הזה ותן דוח קצר בעברית.

כותרת: {title}
תוכן הדף: {page_text}

החזר בפורמט:
✅ יתרונות: [2-3 יתרונות]
⚠️ לשים לב: [1-2 דברים לבדוק]
📍 מיקום: [תיאור האזור אם ידוע]
💡 המלצה: [משפט אחד — שווה/לא שווה לבדוק]"""
            }]
        )
        return analysis.choices[0].message.content
    except Exception as e:
        return "לא הצלחתי לנתח את הדירה"

def find_top_3(properties: list, city: str, rooms: str,
               budget: int, preferences: dict) -> str:

    if not properties:
        city_slug = CITY_MAP.get(city, city.lower().replace(" ", "-"))
        return f"""לא נמצאו דירות ב{city} עד {budget:,} ₪ כרגע.

הדירות מתעדכנות כל כמה שעות — נסה שוב מאוחר יותר.
🔍 חפש ידנית: https://realta.co.il/en/{city_slug}/"""

    top = properties[:3]
    results = []

    for i, p in enumerate(top):
        print(f"מנתח דירה {i+1}: {p['url']}")
        analysis = deep_analyze_property(p['url'], p['title'])
        results.append({
            "num": i + 1,
            "title": p['title'],
            "price": p['price'],
            "url": p['url'],
            "analysis": analysis
        })

    summary_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=700,
        messages=[{
            "role": "user",
            "content": f"""אתה עוזר נדלן ישראלי מקצועי.

הלקוח מחפש: {rooms} חדרים ב{city} עד {budget:,} ₪

ניתוח 3 דירות:

{chr(10).join([
    f"דירה {r['num']}:{chr(10)}{r['title']}{chr(10)}מחיר: {r['price']:,} ₪{chr(10)}{r['analysis']}{chr(10)}קישור: {r['url']}"
    for r in results
])}

כתוב דוח סיכום ללקוח בעברית בפורמט:

🏠 דירה 1 — [מחיר] ₪
✅ יתרונות: [פרט]
⚠️ לשים לב: [פרט]
📍 מיקום: [פרט]
🔗 [קישור]

🏠 דירה 2 — [מחיר] ₪
✅ יתרונות: [פרט]
⚠️ לשים לב: [פרט]
📍 מיקום: [פרט]
🔗 [קישור]

🏠 דירה 3 — [מחיר] ₪
✅ יתרונות: [פרט]
⚠️ לשים לב: [פרט]
📍 מיקום: [פרט]
🔗 [קישור]

⭐ הכי מומלצת: דירה X כי [סיבה קצרה וברורה]"""
        }]
    )

    return summary_response.choices[0].message.content

def search_and_recommend(city: str, rooms: str, budget: str,
                          preferences: dict = {}) -> str:
    try:
        budget_int = int(str(budget).replace(",", "").replace("₪", "").replace("שקל", "").strip())
    except:
        budget_int = 5000

    properties = scrape_rss(city, rooms, budget_int)
    return find_top_3(properties, city, rooms, budget_int, preferences)
