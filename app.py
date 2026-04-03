import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from groq import Groq
from dotenv import load_dotenv

# טעינת משתני סביבה (לקרוסר מה-.env, ל-Hugging Face מה-Secrets)
load_dotenv()

# הגדרת המפתחות - שים לב שהשמות תואמים למה שהגדרת ב-Secrets
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# אתחול הלקוח של Groq
client = Groq(api_key=GROQ_API_KEY)

# הגדרת לוגים (חשוב מאוד כדי לראות שגיאות ב-Hugging Face)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """פונקציה שמטפלת בכל הודעה שנשלחת לבוט"""
    user_text = update.message.text
    
    if not user_text:
        return

    try:
        # שליחת השאלה למודל Llama 3 של Groq
        completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": user_text}
            ],
            model="llama3-8b-8192", # מודל מהיר וחינמי
        )
        
        # חילוץ התשובה
        bot_response = completion.choices[0].message.content
        
        # שליחה חזרה לטלגרם
        await update.message.reply_text(bot_response)
        
    except Exception as e:
        logging.error(f"שגיאה בתקשורת עם Groq: {e}")
        await update.message.reply_text("מצטער, הייתה לי תקלה קטנה במוח. נסה שוב?")

if __name__ == '__main__':
    # בדיקה שהמפתחות הוגדרו
    if not TELEGRAM_TOKEN or not GROQ_API_KEY:
        print("Error: Missing TELEGRAM_TOKEN or GROQ_API_KEY environment variables!")
    else:
        # בניית האפליקציה של טלגרם
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        
        # הגדרה שהבוט יגיב לכל הודעת טקסט (שהיא לא פקודה)
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        
        print("The Bot is alive! Send a message in Telegram.")
        app.run_polling()