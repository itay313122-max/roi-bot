import asyncio
import json
import re
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Tuple
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
from telegram.ext import ContextTypes

from excel_reporter import save_to_excel

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

MIN_IMAGES = 1
MAX_IMAGES = 5
VAT_RATE = 0.18
FRESH_HOURS = 2

def google_maps_url(address: str) -> str:
    q = (address or "ישראל").strip()
    return f"https://www.google.com/maps/search/?api=1&query={quote(q)}"


def generate_facebook_search_url(city: str, budget: str = "", rooms: str = "") -> str:
    """Generate Facebook search URL for apartment groups in the city.
    
    Examples:
    - "תל אביב" → Facebook search for "דירות בתל אביב"
    - With budget/rooms filters for additional context
    """
    search_terms = f"דירות ב{city}"
    if rooms and rooms != "0":
        search_terms += f" {rooms} חדרים"
    if budget:
        search_terms += f" {budget}"
    
    # Facebook search URL for groups and marketplace
    encoded_query = quote(search_terms)
    return f"https://www.facebook.com/search/groups/?q={encoded_query}"


def split_analysis_and_bottom_line(text: str) -> tuple[str, str]:
    """מפריד טקסט ניתוח משורת המחץ (אם הוגדרה)."""
    m = re.search(r"שורת\s*המחץ\s*:\s*(.+)", text, re.DOTALL)
    if m:
        rest = m.group(1).strip()
        first_line = rest.split("\n")[0].strip()
        clean = text[: m.start()].strip()
        return clean, first_line
    return text.strip(), ""


def parse_item_pub_date(item) -> datetime | None:
    raw = item.findtext("pubDate")
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw.strip())
    except Exception:
        return None


def is_fresh_ad(pub_date: datetime | None) -> bool:
    if not pub_date:
        return False
    try:
        pd = pub_date
        if pd.tzinfo is None:
            pd = pd.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - pd.astimezone(timezone.utc)).total_seconds() <= FRESH_HOURS * 3600
    except Exception:
        return False


def detect_publisher_type(html: str) -> str:
    if not html:
        return "לא ידוע"
    if any(
        x in html
        for x in (
            "תיווך",
            "מתווך",
            "סוכנות",
            'משרד הנדל"ן',
            "נדל״ן",
            "broker",
        )
    ):
        return "תיווך"
    if any(x in html for x in ("מפרסם פרטי", "מודעה פרטית", "בעל הדירה", "ללא תיווך")):
        return "פרטי"
    low = html.lower()
    if "private" in low and "agency" not in low:
        return "פרטי"
    return "לא ידוע"


def broker_fee_line(price: int, listing_type: str | None, publisher: str) -> str:
    if publisher != "תיווך" or not price:
        return ""
    lt = listing_type or ""
    is_rent = lt == "השכרה" or (lt != "מכירה" and price < 500_000)
    if is_rent:
        fee = int(round(price * (1 + VAT_RATE)))
        return f"• דמי תיווך משוערים: חודש שכירות + מע״מ ≈ {fee:,} ₪"
    fee = int(round(price * 0.02 * (1 + VAT_RATE)))
    return f"• דמי תיווך משוערים: 2% + מע״מ ≈ {fee:,} ₪"


def extract_sqm_floor_neighborhood(html: str, title: str, city: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser") if html else None
    text = soup.get_text(" ", strip=True) if soup else title
    sqm = ""
    m = re.search(
        r"(\d{2,4})\s*מ[\"״]?\s*ר|מ[\"״]?\s*ר\s*[׳']?\s*(\d{2,4})",
        text,
        re.I,
    )
    if m:
        sqm = (m.group(1) or m.group(2) or "").strip()
    floor = ""
    m2 = re.search(
        r"קומה\s*[:\s]*(\d+)|קומה\s+רגילה\s+(\d+)|(\d+)\s*מתוך\s*\d+\s*קומות",
        text,
        re.I,
    )
    if m2:
        floor = next(g for g in m2.groups() if g)
    neighborhood = ""
    if "—" in title:
        neighborhood = title.split("—")[0].strip()[:80]
    elif "," in title:
        neighborhood = title.split(",")[0].strip()[:80]
    if not neighborhood:
        neighborhood = city
    return {
        "sqm": sqm or "לא צוין",
        "floor": floor or "לא צוין",
        "neighborhood": neighborhood,
    }


def _pick_largest_from_srcset(srcset: str) -> str | None:
    best_url = None
    best_w = -1
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.rsplit(None, 1)
        url = bits[0].strip()
        w = 0
        if len(bits) > 1 and bits[1].endswith("w"):
            try:
                w = int(bits[1][:-1])
            except ValueError:
                pass
        if w >= best_w:
            best_w = w
            best_url = url
    return best_url


def _upgrade_image_url(url: str) -> str:
    """מנסה להחליף גרסאות קטנות בכתובות נפוצות של CDN."""
    if not url:
        return url
    u = url
    for pat, repl in (
        (r"/w_\d+/", "/w_1200/"),
        (r"width=\d+", "width=1200"),
        (r"-\d+x\d+\.(jpg|jpeg|png)", r"-1200x900.\1"),
    ):
        u2 = re.sub(pat, repl, u, flags=re.I)
        if u2 != u:
            u = u2
            break
    return u


def parse_price(text: str) -> int:
    """מחלץ מחיר מכירה (מאות אלפים ומעלה) או השכרה (אלפים) מכותרת/טקסט."""
    nums: List[int] = []
    for n in re.findall(r"\d+", text.replace(",", "").replace("'", "")):
        try:
            nums.append(int(n))
        except ValueError:
            continue
    for val in nums:
        if 300_000 <= val <= 100_000_000:
            return val
    for val in nums:
        if 1_000 <= val <= 299_999:
            return val
    return 999_999_999


def parse_budget_int(budget: str, listing_type: str) -> int:
    raw = str(budget)
    s = raw.replace(",", "").replace("₪", "").replace("שקל", "").strip()
    if "מיליון" in raw or "מילי" in raw:
        m = re.search(r"(\d+\.?\d*)", s)
        if m:
            return int(float(m.group(1)) * 1_000_000)
    m2 = re.search(r"(\d+\.?\d*)", s)
    if m2:
        try:
            v = float(m2.group(1))
            if listing_type == "מכירה" and v < 200_000:
                return int(v * 1_000_000)
            return int(v)
        except ValueError:
            pass
    return 5_000_000 if listing_type == "מכירה" else 8_000


def parse_rooms(text: str) -> float:
    match = re.search(r"(\d+\.?\d*)\s*room", text.lower())
    if match:
        return float(match.group(1))
    return 0


def _matches_listing_type(blob: str, listing_type: str | None) -> bool:
    if not listing_type:
        return True
    b = blob.lower()
    has_rent = any(
        x in blob for x in ("השכרה", "שכירות", "להשכרה")
    ) or "rent" in b or "lease" in b
    has_sale = any(x in blob for x in ("מכירה", "למכירה")) or "sale" in b
    if listing_type == "השכרה":
        if has_sale and not has_rent:
            return False
        return True
    if listing_type == "מכירה":
        if has_rent and not has_sale:
            return False
        return True
    return True


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


def extract_address_from_html(html: str, title: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(attrs={"itemprop": re.compile("streetAddress|addressLocality", re.I)}):
        t = tag.get_text(separator=" ", strip=True)
        if 5 < len(t) < 220:
            return t
    for tag in soup.find_all(class_=re.compile(r"address|location|street", re.I)):
        t = tag.get_text(separator=" ", strip=True)
        if 8 < len(t) < 220:
            return t
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        c = og["content"].strip()
        if 10 < len(c) < 200:
            return c
    if "—" in title:
        return title.split("—")[0].strip()[:180]
    return (title or "כתובת לא צוינה")[:180]


def extract_property_image_urls(page_url: str) -> List[str]:
    """Extract direct image URLs (.jpg, .png, .gif, .webp) from property page."""
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return []
    urls: List[str] = []
    for img in soup.find_all("img"):
        srcset = (img.get("srcset") or "").strip()
        if srcset:
            picked = _pick_largest_from_srcset(srcset)
            if picked:
                src = picked
            else:
                src = ""
        else:
            src = (img.get("src") or img.get("data-src") or img.get("data-lazy-src") or "").strip()
        if not src or src.startswith("data:"):
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = urljoin(page_url, src)
        src = _upgrade_image_url(src)
        low = src.lower()
        # Filter out non-image URLs and UI elements
        if any(x in low for x in ("logo", "icon", "avatar", "pixel", "spacer", "favicon")):
            continue
        # Only accept direct image URLs ending in image extensions
        if not any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp")):
            continue
        if not src.startswith("http"):
            continue
        urls.append(src)
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= MAX_IMAGES:
            break
    return out


def collect_property_images(property_url: str, rss_image: str) -> List[str]:
    """Collect property images from RSS feed and page. Only include direct image URLs."""
    merged: List[str] = []
    
    # Add RSS image if it's a direct image URL
    if rss_image and rss_image.strip():
        rss_url = rss_image.strip()
        rss_low = rss_url.lower()
        if any(rss_low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp")):
            merged.append(rss_url)
    
    # Extract images from property page
    try:
        from_page = extract_property_image_urls(property_url)
        for u in from_page:
            if u not in merged:
                merged.append(u)
    except Exception as page_err:
        print(f"⚠️ שגיאה בחילוץ תמונות מהעמוד: {page_err}")
    
    merged = merged[:MAX_IMAGES]
    if len(merged) >= MIN_IMAGES:
        return merged[:MAX_IMAGES]
    return merged


def scrape_rss(
    city: str,
    rooms: str,
    budget: int,
    listing_type: str | None = None,
) -> list:
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
            pub_date = parse_item_pub_date(item)

            if city_slug not in link.lower():
                continue

            blob = f"{title} {desc} {link}"
            if not _matches_listing_type(blob, listing_type):
                continue

            price = parse_price(title + " " + desc)

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
                "pub_date": pub_date,
            })

        properties.sort(key=lambda x: x["price"])
        print(f"נמצאו {len(properties)} דירות מתאימות")
        return properties

    except Exception as e:
        print(f"שגיאה בסריקה: {e}")
        return []


def deep_analyze_property(
    url: str,
    title: str,
    buyer_profile: str = "קונה/שוכר",
) -> Tuple[str, Dict[str, int]]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)[:4000]

        if buyer_profile == "משקיע":
            focus = (
                "התמקד בתשואה (Yield/ROI), ביקוש להשכרה, סיכונים ועלויות אחזקה — בעברית בלבד."
            )
        else:
            focus = (
                "התמקד בשקט מגורים, קומה, תחבורת ציבור, קרבה לסופרמרקטים ושירותים — בעברית בלבד."
            )

        analysis = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=700,
            messages=[{
                "role": "user",
                "content": f"""נתח את דף הנכס וכתוב דוח בעברית בלבד. קצר וברור.

כותרת: {title}
פרופיל לקוח: {buyer_profile}
{focus}

תוכן הדף:
{page_text}

כתוב רק עם תבליטים (שורה שמתחילה ב-•). חובה לכלול:
• מפת רעשים (הערכת רמת רעש צפויה לפי סוג רחוב/מיקום אם מופיע בדף; אם לא — "לא צוין בדף")
• תחבורה ציבורית
• קרבה לסופרמרקטים / שירותים
• מצב הנכס והאזור (בנייה/שיפוץ אם מוזכר)
{"• תשואה/ROI והתאמה למשקיע" if buyer_profile == "משקיע" else "• התאמה למגורים (שקט, קומה, סביבה)"}

אל תכתוב ציונים מספריים בגוף הטקסט.

בשורה נפרדת אחרי התבליטים, כתוב בדיוק:
שורת המחץ: <משפט אחד שמסכם הכי חשוב>

אחרי שורת המחץ, הוסף בלוק JSON בלבד (בלי טקסט נוסף אחריו):
<<<SCORES>>>
{{"price": X, "location": X, "condition": X, "overall": X}}
<<<END>>>

כאשר X ציון 1–10 ל: מחיר מול שוק, מיקום, מצב, וציון כולל."""
            }]
        )
        raw = analysis.choices[0].message.content or ""
        return _strip_scores_block(raw)
    except Exception:
        return "לא הצלחתי לנתח את הדירה", _default_scores()


def deep_analyze_property_full(
    url: str,
    title: str,
    buyer_profile: str = "קונה/שוכר",
) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)[:8000]

        investor_extra = ""
        if buyer_profile == "משקיע":
            investor_extra = (
                "\n• תשואה (ROI) — השכרה צפויה מול מחיר, סיכונים, השוואה גסה לאזור (רק לפי הדף)\n"
                "• נקודות למשקיע — חוזים, ביקוש, ועד בית אם מוזכר"
            )

        analysis = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1400,
            messages=[{
                "role": "user",
                "content": f"""נתח לעומק את דף הנכס וכתוב דוח מלא בעברית בלבד. השתמש בתבליטים (•) לכל סעיף.

כותרת: {title}
פרופיל לקוח: {buyer_profile}
תוכן הדף: {page_text}

חובה לכלול (בעברית, תבליטים):
• סיכום הנכס — חדרים, קומה, גודל אם מופיע
• מפת רעשים — הערכת רעש לפי מיקום/רחוב (אם אין מידע: "לא צוין בדף")
• תחבורה ציבורית — נגישות והערכה
• קרבה לסופרמרקטים ושירותים
• האזור למגורים — שקט/סואן לפי מה שבדף
• בנייה/שיפוצים/פרויקטים בסביבה אם רלוונטי
• יחס מחיר־ערך מול שוק
• יתרונות וחסרונות
{"• תשואה, סיכונים והתאמה למשקיע — מפורט" if buyer_profile == "משקיע" else "• התאמה למגורים — קומה, שקט, סביבה"}
• המלצה — האם לבצע ביקור או משא ומתן
{investor_extra}

בסוף הדוח, בשורה נפרדת:
שורת המחץ: <משפט אחד>

אל תמציא עובדות שלא בדף. כתוב בטון מקצועי וידידותי."""
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
    analysis_raw: str,
    scores: Dict[str, int],
    address: str,
    fresh: bool,
    publisher: str,
    fee_line: str,
) -> str:
    analysis_clean, bottom = split_analysis_and_bottom_line(analysis_raw)
    fresh_line = "🆕 חדש מהתנור\n\n" if fresh else ""
    fee = f"{fee_line}\n" if fee_line else ""
    msg = (
        f"{fresh_line}"
        f"🏠 דירה {num}\n"
        f"• כותרת: {title}\n"
        f"• כתובת: {address}\n"
        f"• מחיר: {price:,} ₪\n"
        f"• סוג מפרסם: {publisher}\n"
        f"{fee}"
        f"\n{analysis_clean}\n\n"
        f"📊 ציון (1–10):\n"
        f"• מחיר מול שוק: {scores['price']}\n"
        f"• מיקום: {scores['location']}\n"
        f"• מצב: {scores['condition']}\n"
        f"• ציון כולל: {scores['overall']}\n"
    )
    if bottom:
        msg += f"\n\n💬 שורת המחץ: {bottom}"
    msg += f"\n\n🔗 {url}"
    return msg


def _comparison_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| מס׳ דירה | מחיר (₪) | ציון כולל | המלצה |",
        "|-----------|----------|-----------|--------|",
    ]
    for r in rows:
        rec = "מומלץ" if r["overall"] >= 7 else "פחות מתאים"
        lines.append(
            f"| {r['num']} | {r['price']:,} | {r['overall']}/10 | {rec} |"
        )
    return "\n".join(lines)


async def _send_bot_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
):
    chat_id = update.effective_chat.id
    return await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=reply_markup,
    )


async def find_top_3(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    properties: list,
    city: str,
    rooms: str,
    budget: int,
    preferences: dict,
    listing_type: str | None,
    buyer_profile: str,
    user_name: str,
) -> None:
    if not properties:
        city_slug = CITY_MAP.get(city, city.lower().replace(" ", "-"))
        msg = (
            f"לא נמצאו דירות ב{city} במסגרת התקציב והסינון שבחרת.\n\n"
            f"הדירות מתעדכנות כל כמה שעות — נסה שוב מאוחר יותר.\n"
            f"🔍 חיפוש ידני: https://realta.co.il/en/{city_slug}/"
        )
        await _send_bot_text(update, context, msg)
        return

    top = properties[:3]
    results: List[Dict[str, Any]] = []
    excel_rows: List[Dict[str, Any]] = []
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")

    for i, p in enumerate(top):
        print(f"מנתח דירה {i+1}: {p['url']}")
        try:
            page_html = requests.get(p["url"], headers=HEADERS, timeout=15).text
        except Exception:
            page_html = ""
        address = extract_address_from_html(page_html, p["title"]) if page_html else p["title"][:120]
        maps_url = google_maps_url(address)
        publisher = detect_publisher_type(page_html) if page_html else "לא ידוע"
        fresh = is_fresh_ad(p.get("pub_date"))
        meta = extract_sqm_floor_neighborhood(page_html, p["title"], city) if page_html else {
            "sqm": "לא צוין",
            "floor": "לא צוין",
            "neighborhood": city,
        }
        fee_txt = broker_fee_line(p["price"], listing_type, publisher)

        analysis, scores = await asyncio.to_thread(
            deep_analyze_property,
            p["url"],
            p["title"],
            buyer_profile,
        )
        image_urls = collect_property_images(p["url"], p.get("photo_url") or p.get("image") or "")

        entry = {
            "num": i + 1,
            "title": p["title"],
            "price": p["price"],
            "url": p["url"],
            "analysis": analysis,
            "scores": scores,
            "photo_url": p.get("photo_url") or "",
            "image_urls": image_urls,
            "address": address,
            "maps_url": maps_url,
            "publisher": publisher,
            "fresh": fresh,
            "meta": meta,
        }
        results.append(entry)

        body = _format_property_message(
            entry["num"],
            entry["title"],
            entry["price"],
            entry["url"],
            analysis,
            scores,
            address,
            fresh,
            publisher,
            fee_txt,
        )
        # Create navigation keyboard with maps and Facebook search buttons
        fb_search_url = generate_facebook_search_url(city, str(p["price"]), rooms)
        nav_kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📍 נווט לדירה", url=maps_url)],
            [InlineKeyboardButton("🔍 חפש גם בפייסבוק", url=fb_search_url)],
        ])
        await _send_bot_text(update, context, body, reply_markup=nav_kb)

        photos = entry["image_urls"]
        chat_id = update.effective_chat.id
        to_send = photos[:MAX_IMAGES]
        if len(to_send) >= 2:
            try:
                media = [InputMediaPhoto(media=u) for u in to_send]
                await context.bot.send_media_group(chat_id=chat_id, media=media)
            except Exception as ex:
                print(f"שליחת אלבום תמונות נכשלה: {ex}")
                for u in to_send:
                    try:
                        await context.bot.send_photo(chat_id=chat_id, photo=u)
                    except Exception as e2:
                        print(f"תמונה בודדת נכשלה: {e2}")
        elif len(to_send) == 1:
            try:
                await context.bot.send_photo(chat_id=chat_id, photo=to_send[0])
            except Exception as ex:
                print(f"שליחת תמונה נכשלה: {ex}")

        lt_label = listing_type or "לא ידוע"
        excel_rows.append({
            "date": now_str,
            "type": lt_label,
            "neighborhood": meta.get("neighborhood", city),
            "address": address,
            "price": entry["price"],
            "sqm": meta.get("sqm", ""),
            "floor": meta.get("floor", ""),
            "ai_score": scores["overall"],
            "publisher_type": "פרטי" if publisher == "פרטי" else ("תיווך" if publisher == "תיווך" else publisher),
            "url": entry["url"],
        })

    try:
        save_to_excel(excel_rows, user_name)
    except Exception as e:
        print(f"שמירה לאקסל נכסים נכשלה: {e}")

    table_rows = [
        {"num": r["num"], "price": r["price"], "overall": r["scores"]["overall"]}
        for r in results
    ]
    table_msg = "📋 טבלת השוואה בין הדירות\n\n" + _comparison_table(table_rows)
    await _send_bot_text(update, context, table_msg)

    best = max(results, key=lambda x: x["scores"]["overall"])
    await _send_bot_text(
        update,
        context,
        f"⭐ הדירה עם הציון הגבוה ביותר כרגע: דירה {best['num']} "
        f"({best['scores']['overall']}/10).",
    )

    context.user_data["last_properties"] = [
        {
            "num": r["num"],
            "title": r["title"],
            "url": r["url"],
            "price": r["price"],
            "address": r.get("address", ""),
            "buyer_profile": buyer_profile,
        }
        for r in results
    ]
    context.user_data["awaiting_property_choice"] = True
    await _send_bot_text(
        update,
        context,
        "איזו דירה הכי מעניינת אותך? אני יכול לחלוץ עוד פרטים עליה — כתוב 1, 2 או 3.",
    )


async def search_and_recommend(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    city: str,
    rooms: str,
    budget: str,
    preferences: dict | None = None,
    listing_type: str | None = None,
    buyer_profile: str = "קונה/שוכר",
    user_name: str = "",
) -> None:
    if preferences is None:
        preferences = {}
    lt = listing_type or context.user_data.get("listing_type")
    bp = buyer_profile or context.user_data.get("buyer_profile") or "קונה/שוכר"
    budget_int = parse_budget_int(str(budget), lt or "השכרה")

    properties = await asyncio.to_thread(scrape_rss, city, rooms, budget_int, lt)
    await find_top_3(
        update,
        context,
        properties,
        city,
        rooms,
        budget_int,
        preferences,
        lt,
        bp,
        user_name or (update.effective_user.first_name or "משתמש"),
    )
