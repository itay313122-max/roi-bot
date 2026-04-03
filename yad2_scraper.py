import asyncio
import json
import re
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from telegram import Update
from telegram.ext import ContextTypes

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

_SCORES_BLOCK = re.compile(
    r"<<<SCORES>>>\s*(\{.*?\})\s*<<<END>>>", re.DOTALL
)


def parse_price(text: str) -> int:
    numbers = re.findall(r'\d+', text.replace(",", ""))
    for n in numbers:
        try:
            val = int(n)
            if 1000 < val < 100000:
                return val
        except Exception:
            continue
    return 999999


def parse_rooms(text: str) -> float:
    match = re.search(r'(\d+\.?\d*)\s*room', text.lower())
    if match:
        return float(match.group(1))
    return 0


def _default_scores() -> Dict[str, int]:
    return {"price": 5, "location": 5, "condition": 5, "overall": 5}


def _strip_scores_block(text: str) -> Tuple[str, Dict[str, int]]:
    m = _SCORES_BLOCK.search(text)
    if not m:
        return text.strip(), _default_scores()
    try:
        raw = json.loads(m.group(1))
        scores = {
            "price": int(raw.get("price", 5)),
            "location": int(raw.get("location", 5)),
            "condition": int(raw.get("condition", 5)),
            "overall": int(raw.get("overall", 5)),
        }
        for k in scores:
            scores[k] = max(1, min(10, scores[k]))
    except Exception:
        scores = _default_scores()
    clean = _SCORES_BLOCK.sub("", text).strip()
    return clean, scores


def scrape_rss(city: str, rooms: str, budget: int) -> list:
    url = "https://realta.co.il/feed.xml"
    print("סורק RSS של Realta...")

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        root = ET.fromstring(response.content)

        try:
            rooms_num = float(str(rooms).replace("חדרים", "").replace("חדר", "").strip())
        except Exception:
            rooms_num = 3

        city_slug = CITY_MAP.get(city, city.lower().replace(" ", "-"))

        properties = []
        items = root.findall(".//item")
        print(f"סה״כ פריטים ב-RSS: {len(items)}")

        for item in items:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            desc = item.findtext("description") or ""
            enclosure = item.find("enclosure")
            enclosure_url = (enclosure.get("url") or "").strip() if enclosure is not None else ""

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
                "image": enclosure_url,
                "photo_url": enclosure_url,
            })

        properties.sort(key=lambda x: x["price"])
        print(f"נמצאו {len(properties)} דירות מתאימות")
        return properties

    except Exception as e:
        print(f"שגיאה בסריקה: {e}")
        return []


def deep_analyze_property(url: str, title: str) -> Tuple[str, Dict[str, int]]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)[:4000]

        analysis = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""נתח את דף הדירה ותן דוח בעברית (קצר אך ממוקד).

כותרת: {title}
תוכן הדף: {page_text}

כלול בניתוח במפורש (אם אין מידע — כתוב "לא צוין בדף"):
- קרבה לתחבורה ציבורית
- קרבה לסופרמרקט / מרכז מסחרי
- האם האזור נשמע שקט או סואן (לפי תיאור הרחוב/השכונה אם מופיע)
- האם יש בנייה או שיפוץ בסביבה (אם מוזכר)
- מה רמת הדירה ביחס למחיר השוק באזור (הערכה זהירה לפי הנתונים בדף)

אל תכתוב ציונים בטקסט החופשי — רק ניתוח.

בסוף כל הטקסט, הוסף בלוק JSON בדיוק כך (בלי טקסט נוסף אחריו):
<<<SCORES>>>
{{"price": X, "location": X, "condition": X, "overall": X}}
<<<END>>>"""
            }]
        )
        raw = analysis.choices[0].message.content or ""
        return _strip_scores_block(raw)
    except Exception:
        return "לא הצלחתי לנתח את הדירה", _default_scores()


def deep_analyze_property_full(url: str, title: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)[:8000]

        analysis = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1200,
            messages=[{
                "role": "user",
                "content": f"""נתח לעומק את דף הדירה הבא וכתוב דוח מלא ומפורט בעברית ללקוח.

כותרת: {title}
תוכן הדף: {page_text}

הדוח חייב לכלול:
1. סיכום הדירה (חדרים, קומה, גודל אם מופיע)
2. קרבה לתחבורה ציבורית — פירוט והערכה
3. קרבה לסופרמרקט / מרכז מסחרי
4. האם האזור שקט או סואן — והשפעה על מגורים
5. בנייה, שיפוצים או פרויקטים בסביבה (אם רלוונטי)
6. יחס מחיר־ערך מול שוק האזור
7. יתרונות וחסרונות ברורים
8. המלצה מסכמת — האם כדאי לבצע ביקור / משא ומתן

כתוב בטון מקצועי וידידותי. אל תמציא עובדות שלא מופיעות בדף — ציין כשחסר מידע."""
            }]
        )
        return analysis.choices[0].message.content or "לא התקבל ניתוח."
    except Exception as e:
        return f"לא הצלחתי להפיק דוח מלא: {e}"


def _format_property_message(
    num: int,
    title: str,
    price: int,
    url: str,
    analysis_clean: str,
    scores: Dict[str, int],
) -> str:
    return (
        f"🏠 דירה {num} — {title}\n"
        f"💰 {price:,} ₪\n\n"
        f"{analysis_clean}\n\n"
        f"📊 ציון מפורט:\n"
        f"מחיר: {scores['price']}/10\n"
        f"מיקום: {scores['location']}/10\n"
        f"מצב: {scores['condition']}/10\n"
        f"ציון כולל: {scores['overall']}/10\n\n"
        f"🔗 {url}"
    )


def _comparison_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| דירה | מחיר | ציון | המלצה |",
        "|------|------|------|--------|",
    ]
    for r in rows:
        rec = "✅" if r["overall"] >= 7 else "❌"
        lines.append(
            f"| {r['num']}    | {r['price']:,}  | {r['overall']}/10 | {rec}      |"
        )
    return "\n".join(lines)


async def find_top_3(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    properties: list,
    city: str,
    rooms: str,
    budget: int,
    preferences: dict,
) -> None:
    if not properties:
        city_slug = CITY_MAP.get(city, city.lower().replace(" ", "-"))
        msg = (
            f"לא נמצאו דירות ב{city} עד {budget:,} ₪ כרגע.\n\n"
            f"הדירות מתעדכנות כל כמה שעות — נסה שוב מאוחר יותר.\n"
            f"🔍 חפש ידנית: https://realta.co.il/en/{city_slug}/"
        )
        await update.message.reply_text(msg)
        return

    top = properties[:3]
    results: List[Dict[str, Any]] = []

    for i, p in enumerate(top):
        print(f"מנתח דירה {i+1}: {p['url']}")
        analysis, scores = await asyncio.to_thread(
            deep_analyze_property, p["url"], p["title"]
        )
        entry = {
            "num": i + 1,
            "title": p["title"],
            "price": p["price"],
            "url": p["url"],
            "analysis": analysis,
            "scores": scores,
            "photo_url": p.get("photo_url") or p.get("image") or "",
        }
        results.append(entry)

        body = _format_property_message(
            entry["num"],
            entry["title"],
            entry["price"],
            entry["url"],
            analysis,
            scores,
        )
        await update.message.reply_text(body)

        photo = (entry.get("photo_url") or "").strip()
        if photo:
            try:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=photo,
                )
            except Exception as ex:
                print(f"שליחת תמונה נכשלה: {ex}")

    table_rows = [
        {"num": r["num"], "price": r["price"], "overall": r["scores"]["overall"]}
        for r in results
    ]
    table_msg = (
        "📋 טבלת השוואה\n\n"
        + _comparison_table(table_rows)
    )
    await update.message.reply_text(table_msg)

    best = max(results, key=lambda x: x["scores"]["overall"])
    await update.message.reply_text(
        f"⭐ הדירה עם הציון הגבוה ביותר כרגע: דירה {best['num']} "
        f"({best['scores']['overall']}/10)."
    )

    context.user_data["last_properties"] = [
        {
            "num": r["num"],
            "title": r["title"],
            "url": r["url"],
            "price": r["price"],
        }
        for r in results
    ]
    context.user_data["awaiting_property_choice"] = True
    await update.message.reply_text(
        "איזו דירה הכי מעניינת אותך? אני יכול לחפש עוד פרטים עליה — "
        "כתוב 1, 2 או 3."
    )


async def search_and_recommend(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    city: str,
    rooms: str,
    budget: str,
    preferences: dict | None = None,
) -> None:
    if preferences is None:
        preferences = {}
    try:
        budget_int = int(str(budget).replace(",", "").replace("₪", "").replace("שקל", "").strip())
    except Exception:
        budget_int = 5000

    properties = await asyncio.to_thread(scrape_rss, city, rooms, budget_int)
    await find_top_3(
        update, context, properties, city, rooms, budget_int, preferences
    )
