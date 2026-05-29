# FPV DRONE DETECTOR — מדריך מלא

> מערכת זיהוי רחפנים אקוסטי + ויזואלי | גרסה 3.0  
> קובץ זה מסביר הכל — מהתקנה עד ניתוח שדה

---

## תוכן עניינים

1. [מה המערכת הזו עושה](#1-מה-המערכת-הזו-עושה)
2. [איך זה עובד — הסבר טכני פשוט](#2-איך-זה-עובד)
3. [התקנה מלאה](#3-התקנה-מלאה)
4. [הפעלה ראשונה — בית](#4-הפעלה-ראשונה)
5. [הפעלה בשטח — פרוטוקול מלא](#5-הפעלה-בשטח)
6. [קריאת המסך — כל שדה מוסבר](#6-קריאת-המסך)
7. [קריאת ה-CSV — ניתוח תוצאות](#7-קריאת-ה-csv)
8. [טבלת תדרים — כל סוגי הרחפנים](#8-טבלת-תדרים)
9. [בעיות נפוצות ופתרונות](#9-בעיות-נפוצות)
10. [הגבלות המערכת](#10-הגבלות-המערכת)

---

## 1. מה המערכת הזו עושה

### הרעיון הבסיסי

הרחפן מסגיר את עצמו בשתי דרכים:
- **הוא מרעיש** — הרוטורים מייצרים קול בתדר מאוד ספציפי
- **הוא נראה** — מצלמה עם בינה מלאכותית יכולה לזהות אותו בשמיים

המערכת עושה שניהם במקביל, ומשלבת את התוצאות:

```
מיקרופון ──→ ניתוח קול ─────┐
                              ├──→ [CONFIRMED] / [ACOUSTIC] / [VISUAL] / [CLEAR]
מצלמה    ──→ YOLOv8 AI  ─────┘
```

### מה היא מספרת לך

| מצב | משמעות |
|-----|--------|
| **CONFIRMED** | גם שמענו וגם ראינו — ביטחון גבוה מאוד |
| **ACOUSTIC ONLY** | שמענו, אבל לא ראינו (מחוץ לשדה ראייה, לילה, ערפל) |
| **VISUAL ONLY** | ראינו, אבל לא שמענו (רחפן חשמלי שקט, רחוק מאוד) |
| **CLEAR** | לא שמענו ולא ראינו — ניטור רגיל |

### הקבצים במערכת

```
drone_detector.py   ← הלב של המערכת (תמיד רץ ברקע)
ui_field.py         ← ממשק שטח בטרמינל (מה שרואים בשדה)
test_logger.py      ← מקליט CSV בשקט, בלי ממשק
README_FIELD.md     ← מדריך קצר לשטח (אנגלית)
FULL_GUIDE.md       ← המדריך הזה
```

---

## 2. איך זה עובד

### שכבה 1 — אקוסטיקה (ניתוח קול)

#### הרוטור כמטרונום

כל רוטור מסתובב במהירות קבועה. כל פעם שהלהב עובר דרך האוויר הוא "מכה" גל לחץ קטן.  
אם הלהב עובר **1,000 פעמים בשנייה** — נשמע **טון של 1,000 הרץ**.

```
RPM (סיבובים לדקה) ÷ 60 × מספר להבים = BPF (הרץ)

דוגמה — FPV 5 אינץ':
25,000 RPM ÷ 60 × 2 להבים = 833 Hz
```

זה נקרא **BPF — Blade Pass Frequency** (תדר מעבר הלהב).  
כמו שכל מכונית יש לה "קול" שונה לפי מנוע — לכל רחפן יש "חתימה תדר" ייחודית.

#### מה המיקרופון שומע

המיקרופון לוקח **1,024 דגימות** של האוויר כל פרק זמן קצר (~64ms).  
אחר כך האלגוריתם מבצע **FFT — Fast Fourier Transform** (טרנספורמט פורייה המהיר).

**מה זה FFT?** דמיינו שאתם מקשיבים לתזמורת.  
אתם שומעים כל המכשירים ביחד — FFT "מפריד" את הצלילים ואומר: "יש כאן כינור ב-440Hz, חצוצרה ב-880Hz, תוף ב-60Hz."

כך FFT מפריד את כל רעשי הרקע ומחפש את הטון הספציפי של הרוטור.

#### SNR — איך יודעים שהאות אמיתי?

**SNR = Signal-to-Noise Ratio** = יחס אות לרעש.

**משל:** אתם בחדר רועש ומישהו לוחש.  
- אם כולם שותקים ומישהו לוחש — אתם שומעים (SNR גבוה)
- אם כולם צועקים ומישהו לוחש — לא שומעים (SNR נמוך)

```
SNR = עוצמת הטון של הרחפן ÷ רעש הרקע הממוצע

SNR < 3×   → מתעלמים (רעש רקע בלבד)
SNR 3–8×   → זיהוי חלש (רחפן רחוק, או רעש גבוה)
SNR > 15×  → זיהוי חזק (רחפן קרוב)
```

#### הרמוניות — עוד ראיה שזה רחפן

כשרוטור מסתובב, הוא לא מייצר רק את ה-BPF.  
הוא מייצר גם **כפולות** שלו: פעמיים, שלוש פעמים, ארבע פעמים...

```
FPV 5" ב-833 Hz יייצר גם:
  833 Hz  ← f₀ (הבסיסי)
 1666 Hz  ← f₀×2
 2499 Hz  ← f₀×3
 3332 Hz  ← f₀×4
```

אם הטון ב-833Hz יש לו "חברים" בכפולות — כמעט בוודאות זה רחפן ולא מכונית/עורב.  
**משל:** כמו שאתם מזהים גיטרה לא רק מהתו הראשי, אלא מהצליל העשיר שלה עם כל ההרמוניות.

#### EMA — מסנן רעש מתמטי

התדר הגולמי קופץ כל פריים.  
**EMA (Exponential Moving Average)** ממצע אותו — נותן משקל גדול לערכים האחרונים, קטן לישנים.

```
ביטחון חדש = 0.25 × ביטחון_פריים + 0.75 × ביטחון_קודם
```

**משל:** כמו שאתם מחליטים אם מישהו עצוב — לא מסתמכים על חיוך אחד, אלא על ממוצע ההתנהגות.

#### Doppler — מתקרב או מתרחק?

כשמשהו מתקרב אליכם — הגלים "מתדחסים" ונשמעים בתדר גבוה יותר.  
כשמתרחק — נשמעים בתדר נמוך יותר.

**משל:** אמבולנס עובר — הסירנה עולה כשמתקרב, יורדת כשמתרחק.

```
f₀ עולה לאורך 10 פריימים → APPROACHING (מתקרב)
f₀ יורדת לאורך 10 פריימים → RECEDING (מתרחק)
f₀ יציבה → STABLE (מרחף)
```

---

### שכבה 2 — ויזואלי (YOLOv8)

**YOLO** (You Only Look Once) הוא אלגוריתם בינה מלאכותית לזיהוי עצמים בתמונה.  
הוא מסתכל על כל הפריים בבת אחת ומסמן תיבות (Bounding Boxes) סביב עצמים שזיהה.

המערכת משתמשת בשני מודלים (לפי מה שזמין):
1. **`keremberke/yolov8n-drone-detection`** — מודל שאומן ספציפית על רחפנים (מ-HuggingFace)
2. **`yolov8n.pt` (COCO)** — מודל כללי, משתמש בקלאס "airplane" כ-proxy לרחפן

**סף זיהוי:** 60% ביטחון — מתחת לזה מתעלמים.

---

### שילוב (Sensor Fusion)

```python
if acoustic_alarm AND visual_alert:  → "CONFIRMED"
if acoustic_alarm only:              → "ACOUSTIC ONLY"
if visual_alert only:                → "VISUAL ONLY"
else:                                → "CLEAR"
```

---

## 3. התקנה מלאה

### דרישות מערכת

- Python 3.11+
- מיקרופון USB (מומלץ) או מובנה
- מצלמה (אופציונלי — אם אין, עובד בסימולציה)
- Linux / macOS / Windows (Linux מועדף בשטח)
- מינימום 4GB RAM (YOLOv8 צורך ~1.5GB)

### שלב 1 — שכפול הקוד

```bash
git clone https://github.com/itay313122-max/roi-bot.git
cd roi-bot
```

### שלב 2 — התקנת ספריות Python

```bash
# ספריות ליבה — חובה
pip install numpy

# אקוסטיקה
# על Linux — קודם portaudio:
sudo apt-get install -y portaudio19-dev
pip install pyaudio

# ויזואלי
pip install ultralytics opencv-python

# ממשק שטח
pip install rich
```

**מה כל ספרייה עושה:**

| ספרייה | תפקיד |
|--------|--------|
| `numpy` | חישובים מתמטיים מהירים (FFT, RMS) |
| `pyaudio` | קריאת אודיו ממיקרופון בזמן אמת |
| `ultralytics` | YOLOv8 — זיהוי רחפנים בתמונה |
| `opencv-python` | קריאת פריימים מהמצלמה |
| `rich` | ממשק טרמינל צבעוני |

### שלב 3 — אימות ההתקנה

```bash
# בדיקת Python
python3 --version   # צריך להיות 3.11+

# בדיקת ספריות
python3 -c "import numpy, pyaudio, ultralytics, cv2, rich; print('הכל תקין')"

# בדיקת מיקרופון
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
print('מיקרופונים שנמצאו:')
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    if d['maxInputChannels'] > 0:
        print(f'  [{i}] {d[\"name\"]}')
p.terminate()
"
```

---

## 4. הפעלה ראשונה

### בפעם הראשונה — כיוון המערכת לסביבה שלכם

#### שלב א' — קליברציה (כיול רעש הסביבה)

לפני שתתחילו לזהות, המערכת צריכה ללמוד מה "שקט" בסביבה שלכם.  
60 שניות ללא רחפנים:

```bash
python drone_detector.py --calibrate
```

הפלט יראה כך:
```
==========================================================
  CALIBRATION MODE
  Measuring ambient noise for 60s.
  Keep area clear of drones during calibration.
==========================================================

  [████████████████████████░░░░░░░░░░░░░░░░]   40/60s  frames: 10,240

  CALIBRATION DONE
  noise floor (median RMS) : 0.000312
  noise σ                  : 0.000089
  recommended rms_gate     : 0.000579

  To apply: set  RMS_GATE = 0.000579  in drone_detector.py
==========================================================
```

פתחו `drone_detector.py` ועדכנו את השורה:
```python
# שורה ~61
RMS_GATE = 0.000579   # ← הערך שקיבלתם מהקליברציה
```

> **למה זה חשוב?**  
> ללא כיול, מחשב בחדר ממוזג יקבל ערך שונה ממחשב בשדה ליד גנרטור.  
> `rms_gate` קובע מה "שקט מספיק" כדי לבדוק קול — מתחתיו המערכת מתעלמת מהפריים.

#### שלב ב' — הפעלת הגלאי

```bash
# טרמינל 1
python drone_detector.py
```

הפלט הצפוי:
```
==========================================================
  DRONE DETECTOR — POC v3
  Acoustic BPF + YOLOv8 visual fusion
==========================================================
  Visual: simulation   [YOLOv8n COCO (airplane proxy)]

  Dashboard: http://localhost:8765

  Acoustic: live microphone
  Ctrl+C to stop

  [░░░░░░░░░░░░░░░░░░░░]   0.0%  [CLEAR]
```

#### שלב ג' — ממשק השטח

```bash
# טרמינל 2
python ui_field.py
```

המסך יתמלא בממשק הצבעוני — ראו סעיף 6 לפירוט.

#### שלב ד' — דשבורד הדפדפן (אופציונלי)

```
http://localhost:8765
```

פתחו בדפדפן לממשק גרפי — שימושי לתצוגה על מסך גדול בעמדת פיקוד.

---

## 5. הפעלה בשטח

### לפני היציאה מהבסיס

```bash
# 1. בדיקת מיקרופון
python3 -c "import pyaudio; p=pyaudio.PyAudio(); print(f'{p.get_device_count()} devices'); p.terminate()"

# 2. בדיקת סוללה/חשמל — המחשב לא יכנס לשינה
# Linux:
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

# 3. בדיקה מהירה (10 שניות)
timeout 10 python drone_detector.py 2>&1 | head -20

# 4. אם יש רחפן בדיקה — בדקו זיהוי ב-30 שניות
python test_logger.py --no-launch --output beit_test.csv
```

### בשטח — פרוטוקול הפעלה

#### שלב 1 — הצבת המיקרופון
- **מיקום:** גובה 1–1.5 מטר, פתוח ל-360 מעלות
- **מרחק מרעש:** לפחות 5 מטר מגנרטורים/רכבים
- **רוח:** השתמשו בFoam windscreen — רוח מעל 15 קמ"ש תיצור false positives

#### שלב 2 — קליברציה בשטח

```bash
# קליברציה קצרה בסביבה החדשה (30 שניות)
python drone_detector.py --calibrate --cal-duration 30
```

> **קריטי:** כל סביבה שונה. המשרד שקט, השדה רועש.  
> קליברציה בבית לא מספיקה לשטח.

אם משתמשים ב-UI:
- הפעילו `drone_detector.py` רגיל
- פתחו `ui_field.py`
- לחצו **C** לקליברציה 30 שניות

#### שלב 3 — הפעלת הקלטת נתונים

```bash
# אפשרות א' — מה-UI: לחצו L
# אפשרות ב' — בטרמינל נפרד:
python test_logger.py --no-launch --output session_$(date +%Y%m%d_%H%M).csv
```

#### שלב 4 — פרשנות התוצאות בשדה

```
CONFIRMED   → פעולה מיידית נדרשת
ACOUSTIC    → סרקו עם עיניים/מצלמה, ייתכן מחוץ לשדה ראייה
VISUAL      → בדקו אם שקט מאוד (חשמלי) או רחוק מאוד
CLEAR       → המשיכו לנטר
```

#### שלב 5 — לאחר הבדיקה

```bash
# העתיקו את ה-CSV למחשב ניתוח
cp field_test_*.csv /path/to/analysis/

# נקו קבצים זמניים
rm -f field_test_*.csv
```

---

## 6. קריאת המסך

### מבנה המסך המלא

```
┌─────────────────────────────────────────────────── UPTIME 00:03:24 ─┐
│  FPV DETECT v2.0 — FIELD MODE       ● field_test_20260529.csv (847) │
├─────────────────────────────────┬───────────────────────────────────┤
│         ACOUSTIC                │         VISUAL (YOLOv8)           │
│  ████████████░░░░░░░░  60%  [A] │  ██████░░░░░░░░░░░░  30%  [B]    │
│  f₀   847 Hz           [C]      │  Model  drone-specific (HF)  [J]  │
│  RPM  ~25,410          [D]      │  FPS    14.2                 [K]  │
│  Type  FPV 5" Racing   [E]      │  Source LIVE CAMERA          [L]  │
│  SNR   8.3×            [F]      │  Alert  no                   [M]  │
├─────────────────────────────────┴───────────────────────────────────┤
│                                                                      │
│              ⚠  ACOUSTIC ONLY              [G]                      │
│                                                                      │
│         Direction : ↑  APPROACHING  (+23.4 Hz/sec)  [H]            │
│         Est. Range: MEDIUM  (50–200m)               [I]             │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  noise floor: 0.000312   rms_gate: 0.000579  frames: 1,847  [N]     │
├─ DETECTION LOG ──────────────────────────────────────────────────────┤
│  TIME     TYPE           f₀       SNR    THREAT         DIR   [O]   │
│  08:42:31 FPV 5" Racing  847 Hz   8.3×  CONFIRMED      APPR        │
│  08:41:15 FPV 5" Racing  831 Hz   6.1×  ACOUSTIC ONLY  STABLE      │
│  08:39:02 DJI Mavic      183 Hz   4.2×  VISUAL ONLY    RECED       │
├──────────────────────────────────────────────────────────────────────┤
│  [C] Calibrate   [L] Stop Log   [R] Reset   [Q] Quit    [P]        │
└──────────────────────────────────────────────────────────────────────┘
```

### פירוט כל שדה

#### חלק עליון — בר מצב

| אות | שדה | משמעות |
|-----|-----|--------|
| — | **UPTIME** | כמה זמן המערכת רצה בלי הפסקה |
| — | **● שם קובץ** | ● אדום = מקליט CSV כעת. מספר = כמה שורות נרשמו |

#### פאנל ACOUSTIC (שמאל)

| אות | שדה | משמעות | דוגמה |
|-----|-----|--------|-------|
| A | **בר ביטחון** | כמה בטוח הגלאי שיש רחפן (0–100%) | `████░░░ 60%` |
| C | **f₀** | התדר הבסיסי שנמצא — חתימת הרוטור | `847 Hz` |
| D | **RPM** | מהירות הסיבוב המשוערת של המנוע | `~25,410` |
| E | **Type** | הפרופיל הקרוב ביותר בבסיס הנתונים | `FPV 5" Racing` |
| F | **SNR** | יחס אות/רעש — "כמה חזק לעומת הרקע" | `8.3×` |

**צבעי הבר:**
- 🟢 ירוק = CLEAR, אין אזעקה
- 🔴 אדום = אזעקה פעילה (confidence > 35% ל-2+ פריימים)

#### פאנל VISUAL (ימין)

| אות | שדה | משמעות | דוגמה |
|-----|-----|--------|-------|
| B | **בר ביטחון** | ביטחון YOLOv8 בגילוי הויזואלי | `██░░░░ 30%` |
| J | **Model** | המודל שנטען | `drone-specific (HF)` |
| K | **FPS** | קצב עיבוד הפריימים | `14.2` |
| L | **Source** | `LIVE CAMERA` = מצלמה אמיתית. `SIMULATION` = אין מצלמה | — |
| M | **Alert** | `YES` = עבר סף 60%. `no` = מתחת לסף | — |

#### פאנל COMBINED THREAT (מרכז)

| אות | שדה | משמעות |
|-----|-----|--------|
| G | **איום משולב** | שילוב שתי השכבות. **CONFIRMED** מהבהב באדום |
| H | **Direction** | כיוון הרחפן לפי שינוי f₀ בזמן. `+23.4 Hz/sec` = עולה במהירות הזו |
| I | **Est. Range** | הערכת טווח מ-SNR. **לא מדויק** — ראו סעיף 10 |

**טבלת צבעים לאיום:**
```
CONFIRMED    → אדום מהבהב  ‼‼
ACOUSTIC     → צהוב        ⚠
VISUAL       → כחול        ●
CLEAR        → ירוק        ✓
```

#### שורת סטטיסטיקות

| אות | שדה | משמעות |
|-----|-----|--------|
| N | **noise floor** | רמת הרעש הבסיסית שנמדדה בקליברציה |
| N | **rms_gate** | הסף שמעליו מעבדים פריים (מתחתיו = "שקט") |
| N | **frames** | סה"כ פריימי אודיו שעובדו מתחילת ההפעלה |

#### לוג זיהויים

| אות | שדה | משמעות |
|-----|-----|--------|
| O | **TIME** | שעה מדויקת שהזיהוי התחיל |
| O | **TYPE** | הפרופיל שזוהה |
| O | **f₀** | התדר שנמדד |
| O | **SNR** | יחס אות/רעש ברגע הזיהוי |
| O | **THREAT** | רמת האיום המשולבת |
| O | **DIR** | APPR / STABLE / RECED |

#### שורת פקודות

| אות | מקש | פעולה |
|-----|-----|-------|
| P | **C** | קליברציה 30 שניות — הצג progress bar במקום פאנל האיום |
| P | **L** | הפעל/כבה הקלטת CSV. שם הקובץ: `field_test_YYYYMMDD_HHMMSS.csv` |
| P | **R** | אפס נתוני הסשן (frames, detections, קליברציה) |
| P | **Q** | יציאה נקייה — שומר ומסגר CSV לפני |

---

## 7. קריאת ה-CSV

### פורמט הקובץ

```csv
timestamp,acoustic_confidence,visual_confidence,combined_threat,f0_hz,rpm_estimate,drone_type,direction,distance_phase
2026-05-29T08:42:31.456,85.3,0.0,ACOUSTIC ONLY,847,25410,FPV 5" Racing (Kamikaze),APPROACHING,INBOUND
2026-05-29T08:42:31.956,91.2,73.0,CONFIRMED,851,25530,FPV 5" Racing (Kamikaze),APPROACHING,INBOUND
2026-05-29T08:42:32.456,78.4,68.0,CONFIRMED,839,25170,FPV 5" Racing (Kamikaze),STABLE,HOVER/OVERHEAD
```

### הסבר כל עמודה

| עמודה | פורמט | הסבר |
|-------|--------|-------|
| `timestamp` | ISO8601 | שעת הרישום המדויקת עד מילישניות |
| `acoustic_confidence` | 0.0–100.0 | ביטחון אקוסטי מ-EMA |
| `visual_confidence` | 0.0–100.0 | ביטחון YOLOv8 |
| `combined_threat` | מחרוזת | CONFIRMED / ACOUSTIC ONLY / VISUAL ONLY / CLEAR |
| `f0_hz` | שלם Hz | התדר הבסיסי. 0 = אין זיהוי |
| `rpm_estimate` | שלם | RPM משוער |
| `drone_type` | מחרוזת | פרופיל שזוהה. — = אין |
| `direction` | מחרוזת | APPROACHING / RECEDING / STABLE |
| `distance_phase` | מחרוזת | INBOUND / OUTBOUND / HOVER/OVERHEAD / N/A |

### איך לנתח — Python

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("field_test_20260529.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp")

# ── 1. סיכום הסשן ──────────────────────────────────────────
total = len(df)
events = df[df["combined_threat"] != "CLEAR"]
print(f"סה\"כ שורות: {total}")
print(f"אירועי זיהוי: {len(events)}")
print(f"\nפירוט לפי סוג:")
print(df["combined_threat"].value_counts())

# ── 2. ציר זמן ביטחון ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
df["acoustic_confidence"].plot(ax=ax1, color="green", label="אקוסטי")
df["visual_confidence"].plot(ax=ax1, color="blue", label="ויזואלי")
ax1.axhline(35, color="orange", linestyle="--", label="סף אזעקה")
ax1.axhline(60, color="red", linestyle="--", label="סף ויזואלי")
ax1.set_ylabel("ביטחון (%)"); ax1.legend(); ax1.set_title("ציר זמן זיהוי")

# ── 3. תדר לאורך זמן ───────────────────────────────────────
df[df["f0_hz"] > 0]["f0_hz"].plot(ax=ax2, color="purple", marker=".", ms=2)
ax2.set_ylabel("f₀ (Hz)"); ax2.set_title("תדר בסיסי — שינוי Doppler")
plt.tight_layout(); plt.show()

# ── 4. סיכום אירועים ────────────────────────────────────────
confirmed = df[df["combined_threat"] == "CONFIRMED"]
if not confirmed.empty:
    print(f"\nאירועי CONFIRMED:")
    print(f"  התחלה: {confirmed.index.min()}")
    print(f"  סיום:  {confirmed.index.max()}")
    print(f"  משך:   {confirmed.index.max() - confirmed.index.min()}")
    print(f"  f₀ ממוצע: {confirmed['f0_hz'].mean():.0f} Hz")
    print(f"  RPM ממוצע: {confirmed['rpm_estimate'].mean():,.0f}")
```

### תוצאה טובה לעומת רעה

**תוצאה טובה — זיהוי FPV ברור:**
```csv
08:42:31,12.3,0.0,CLEAR,0,0,—,STABLE,N/A
08:42:32,45.1,0.0,ACOUSTIC ONLY,847,25410,FPV 5" Racing,APPROACHING,INBOUND
08:42:32,78.3,0.0,ACOUSTIC ONLY,855,25650,FPV 5" Racing,APPROACHING,INBOUND
08:42:33,91.2,73.0,CONFIRMED,863,25890,FPV 5" Racing,APPROACHING,INBOUND
08:42:34,88.6,78.0,CONFIRMED,857,25710,FPV 5" Racing,STABLE,HOVER/OVERHEAD
08:42:35,71.3,65.0,CONFIRMED,831,24930,FPV 5" Racing,RECEDING,OUTBOUND
```
ניתוח: f₀ עולה→ מוצלב→ יציב→ יורד — פלייאובר קלאסי. ✓

**תוצאה רעה — false positives:**
```csv
09:15:11,23.1,0.0,ACOUSTIC ONLY,183,5490,DJI Mavic,STABLE,HOVER/OVERHEAD
09:15:12,18.4,0.0,ACOUSTIC ONLY,183,5490,DJI Mavic,STABLE,HOVER/OVERHEAD
09:15:13,21.7,0.0,ACOUSTIC ONLY,183,5490,DJI Mavic,STABLE,HOVER/OVERHEAD
```
ניתוח: f₀=183Hz **קבוע לחלוטין** + STABLE + אין VISUAL = כנראה HVAC/מאוורר.  
פתרון: שנו `SCAN_MIN_HZ = 200` לסינון תדרים נמוכים.

---

## 8. טבלת תדרים

### כל סוגי הרחפנים

| סוג | BPF (Hz) | RPM | SNR מינימום | רמת איום | איך נשמע |
|-----|----------|-----|------------|----------|----------|
| **FPV 5" Racing (Kamikaze)** | 667–1200 | 20k–36k | 6.0× | 🔴 HIGH | שריקה חדה וגבוהה, "וויז'" — כמו נסירה |
| **FPV 3" Micro (Suicide)** | 833–1500 | 25k–45k | 5.0× | 🔴 HIGH | שריקה עוד יותר גבוהה ודקה |
| **FPV 7" Long Range** | 400–733 | 12k–22k | 5.0× | 🔴 HIGH | אמצע — ין יין עמוק יותר |
| **DJI Mavic (Recon/ISR)** | 150–217 | 4.5k–6.5k | 4.0× | 🟡 MEDIUM | זמזום נמוך, כמו דבורה גדולה |
| **DJI Matrice (Heavy Lift)** | 100–167 | 3k–5k | 3.0× | 🟡 MEDIUM | רעש נמוך ועמוק מאוד |
| **Shahed-136 / Geran-2** | 200–283 | 6k–8.5k | 3.0× | 🔴 CRITICAL | מנוע בנזין — לא עקבי, "טחטח" |
| **Lancet Loitering** | 267–500 | 8k–15k | 4.0× | 🔴 HIGH | חשמלי — שקט יחסית, חד |

### הסבר לכל טור

**BPF (Hz):** הטווח מייצג את הרחפן ב-RPM מינימלי עד מקסימלי.  
דוגמה: FPV 5" ב-20,000 RPM → 667Hz. ב-36,000 RPM → 1,200Hz.

**SNR מינימום:** כמה חזק צריך האות לעומת הרעש כדי לזהות.  
רחפן עם SNR נמוך (3.0×) — DJI Matrice — קשה יותר לזהות כי הוא שקט יחסית.

**איך נשמע:** תיאור האזנה ישירה (ללא ציוד) — שימושי לסינון שגיאות.

### חפיפות בין פרופילים

```
100──────────────────────────────────────────────────────1500 Hz
     DJI Matrice    ████
     DJI Mavic          ██
     Shahed-136          ████
     Lancet                   ████████
     FPV 7"                   ████████████
     FPV 5"                              ████████████████
     FPV 3"                                   █████████████████
```

כשתדר נופל בחפיפה, המערכת משתמשת ב**ציון מרכזיות** — הפרופיל שה-f₀ הנמדד קרוב יותר לאמצע הטווח שלו מנצח.

---

## 9. בעיות נפוצות

### "CONFIRMED" כל הזמן — ללא רחפן

**תסמינים:** confidence > 80%, drone_type מלא, בלי ויזואלי

**צ'ק ליסט:**
```bash
# 1. מה התדר?
curl -s http://localhost:8765/api/state | python3 -c "
import json,sys; d=json.load(sys.stdin)
print(f\"f0={d['f0_hz']}Hz  type={d['profile_label']}\")
"

# 2. האם התדר קבוע לחלוטין? (מאוורר/HVAC)
# → f0 שלא זז כלל = מכשיר חשמלי קבוע, לא רחפן

# 3. הרם את rms_gate ב-2×
# ב-drone_detector.py: RMS_GATE = 0.001  # פי 2 מהערך הנוכחי

# 4. הרם את MIN_SNR
# MIN_SNR = 5.0  # מ-3.0

# 5. אם תדר 183Hz קבוע — כנראה HVAC
# SCAN_MIN_HZ = 220  # דלג על תחום ה-Mavic
```

### המיקרופון לא נמצא

**תסמינים:** `Acoustic: simulation` בלי שביקשתם

```bash
# בדקו מה התקנות:
python3 -c "
import pyaudio; p=pyaudio.PyAudio()
for i in range(p.get_device_count()):
    d=p.get_device_info_by_index(i)
    if d['maxInputChannels']>0:
        print(i, d['name'])
p.terminate()
"

# אם לא נמצא כלום:
# Linux — בדיקת ALSA:
arecord -l

# USB מיקרופון — נסו לנתק ולחבר מחדש
# בדקו dmesg:
dmesg | grep -i audio | tail -5
```

**אם pyaudio לא מותקן בכלל:**
```bash
sudo apt-get install -y portaudio19-dev
pip uninstall pyaudio -y
pip install pyaudio
```

### YOLOv8 לא נטען

**תסמינים:** `Visual: disabled (model load failed)`

```bash
# בדיקת ultralytics:
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('OK')"

# אם נכשל:
pip install --upgrade ultralytics

# אם אין אינטרנט — הורידו ידנית:
# yolov8n.pt מ-GitHub Releases של ultralytics
# שמרו בתיקיית roi-bot/
```

**בסביבת ענן** (אין GPU): המודל ירוץ על CPU, ייתכן איטי (5–10 FPS). נורמלי.

### Confidence נמוך תמיד (< 20%)

**תסמינים:** גלאי אף פעם לא מגיע ל-ALARM

**סיבות אפשריות:**
1. **rms_gate גבוה מדי** — מסנן גם את הרחפן
   ```python
   # הורידו ב-50%:
   RMS_GATE = 0.000289   # חצי מהערך הנוכחי
   ```

2. **מיקרופון רחוק מדי** — הרחפן רחוק מ-200 מטר
   
3. **MIN_SNR גבוה מדי**
   ```python
   MIN_SNR = 2.5   # מ-3.0
   ```

4. **ALARM_FRAMES גבוה מדי** — מחכה יותר פריימים מדי
   ```python
   ALARM_FRAMES = 1   # מ-2
   ```

### "DETECTOR OFFLINE" בממשק

```bash
# בדקו שה-detector רץ:
ps aux | grep drone_detector

# בדקו שהפורט פנוי:
curl http://localhost:8765/api/state

# אם הפורט תפוס:
python drone_detector.py --port 8766
python ui_field.py --port 8766
```

### False positives בגלל רוח

**תסמינים:** confidence מתנדנד 10–35% בתנאי רוח, f₀ משתנה

```bash
# 1. הוסיפו Foam windscreen למיקרופון
# 2. הרימו rms_gate ב-3×
# 3. הרימו ALARM_FRAMES ל-4:
ALARM_FRAMES = 4   # דורש 4 פריימים רצופים לפני אזעקה
```

### False positives מעורבים / מכונית

| מקור | תדר אופייני | איך לסנן |
|------|-------------|----------|
| מאוורר מחשב | 50–200 Hz, קבוע | `SCAN_MIN_HZ = 220` |
| HVAC / מזגן | 100–250 Hz, קבוע | `SCAN_MIN_HZ = 270` |
| מכונית עוברת | רחב-טווח, חולף | הגדילו `ALARM_FRAMES = 4` |
| עורב/ציפור | לא הרמוני | הרמוניות לא יתאמו — פחות בעיה |
| גנרטור | 50/60 Hz, קבוע | `SCAN_MIN_HZ = 120` |
| רוח | אקראי | foam windscreen + `rms_gate × 3` |

---

## 10. הגבלות המערכת

### מה המערכת לא יכולה לעשות

#### 1. טווח — אין ניסים

```
רחפן קטן (FPV)  — זיהוי טוב: עד ~150–200 מטר בתנאים שקטים
                 — זיהוי חלש: 200–350 מטר
                 — לא ניתן:   מעל 350–400 מטר

DJI Mavic       — יותר שקט — זיהוי טוב: עד ~80–120 מטר
Shahed          — מנוע שמע — עד ~300–500 מטר (תלוי רוח)
```

**SNR יורד עם ריבוע המרחק** — פי 2 מרחק = SNR נמוך פי 4.

#### 2. רחפן חשמלי שקט מאוד

FPV 3" מיקרו בגלגל פינפיון שקט יכול להיות מתחת לסף גם ב-50 מטר.  
גרסאות fiber-optic עם מנוע toque quiet עלולות לא להיזהר.  
**פתרון חלקי:** הורידו `MIN_SNR` ו-`rms_gate`, אבל תקבלו יותר false positives.

#### 3. רעש רקע חזק (מעל 85 dBSPL)

- שדה קרב עם ירי: רעש רקע ימסך הכל
- הליקופטר בקרבת מקום: תדרים נמוכים ישבשו זיהוי Mavic/Matrice
- גנרטור גדול: מאסק תדרי BPF נמוכים
- **פתרון:** מיקרופון דירקציונלי (hypercardioid), הצבעה הרחק מהרעש

#### 4. מספר רחפנים בו-זמנית

המערכת מדגמת **פיק אחד בלבד** מה-FFT — הטון החזק ביותר בטווח.  
אם 3 FPV עפים — רק החזק ביותר ייזהה. שאר האות "נטמן".  
**פתרון עתידי:** Multi-peak detection (לא ממומש כרגע).

#### 5. רחפן שמחקה תדר לא בסיס-נתונים

אם רחפן חדש עם תצורת להבים חריגה (5 להבים ב-rpm לא רגיל) — המערכת לא תסווג אותו.  
זיהוי יחסי של תדר יתרחש, אך profile_label יהיה "—".  
**פתרון:** הוסיפו פרופיל ל-`DRONE_DATABASE`.

#### 6. Fiber Optic FPV — בעיה מיוחדת

FPV מוכוון fiber-optic אין לו GPS lock latency — הוא תמיד מרחף נמוך ומהיר.  
הרחפן עצמו נשמע זהה לכל FPV אחר.  
**המערכת תזהה אותו** — אבל לא תדע שהוא fiber-optic.  
זמן תגובה מזיהוי לפגיעה: 2–6 שניות. **יש פחות זמן לפעולה**.

#### 7. אין זיהוי כוונה

המערכת מזהה **נוכחות** של רחפן — לא **כוונה**.  
DJI Mavic של עיתונאי ו-DJI Mavic של ספוטר ייראו זהה.  
לא ניתן להבדיל ביניהם מאקוסטיקה.

#### 8. ממשק הדשבורד לא מאובטח

`http://localhost:8765` — ללא HTTPS, ללא אימות.  
**אין להנגיש לרשת ציבורית** — שרת זה לשימוש מקומי בלבד.

---

## נספח — סיכום פרמטרים לכיוון מהיר

```python
# drone_detector.py — ערכים לכיוון בשטח

# ── כיוון רגישות ──────────────────────────────────────────
RMS_GATE    = 0.0004   # ↑ = פחות רגיש (פחות false pos)
MIN_SNR     = 3.0      # ↑ = דורש אות חזק יותר
ALARM_FRAMES = 2       # ↑ = דורש יציבות יותר פריימים

# ── כיוון Doppler ──────────────────────────────────────────
# ב-DroneDetector._f0_drift():
# rate_hz_per_s > 15 → APPROACHING
# rate_hz_per_s < -15 → RECEDING
# שנו ל-10 לרגישות גבוהה, 25 לרגישות נמוכה

# ── כיוון ממשק ─────────────────────────────────────────────
EMA_ALPHA   = 0.25     # ↓ = חלקה יותר (פחות תנודות)
VISUAL_CONF_THRESHOLD = 0.60  # ↑ = סף גבוה יותר לוויזואלי

# ── סריקת תדרים ────────────────────────────────────────────
SCAN_MIN_HZ = 80       # ↑ = דלג על תדרים נמוכים (HVAC)
SCAN_MAX_HZ = 1600     # ↓ = דלג על תדרים גבוהים
```

---

*מדריך זה נכתב ל-drone_detector.py v3 | הסבר טכני ופרוטוקול שטח*
