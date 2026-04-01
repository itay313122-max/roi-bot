from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import os
from dotenv import load_dotenv
from agent_brain import think, extract_client_info
from excel_reporter import add_client
from yad2_scraper import search_and_recommend

load_dotenv()

conversations = {}
searching = {}
questions_asked = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    user_message = update.message.text

    if user_id not in conversations:
        conversations[user_id] = []
        searching[user_id] = False
        questions_asked[user_id] = 0
        print(f"לקוח חדש: {user_name}")

    conversations[user_id].append({
        "role": "user",
        "content": user_message
    })

    reply = think(conversations[user_id])

    conversations[user_id].append({
        "role": "assistant",
        "content": reply
    })

    # שלב 1 — זיהוי שרועי מוכן לחפש
    if ("מתחיל לחפש" in reply or "אחפש עבורך" in reply) and not searching[user_id]:
        searching[user_id] = True
        questions_asked[user_id] = 0

        await update.message.reply_text(reply)

        try:
            info = extract_client_info(conversations[user_id])

            # שמירה באקסל
            try:
                add_client(
                    name=user_name,
                    area=info.get("area", "לא ידוע"),
                    rooms=info.get("rooms", "לא ידוע"),
                    budget=info.get("budget", "לא ידוע"),
                    deal_type=info.get("deal_type", "לא ידוע"),
                    notes=info.get("notes", "")
                )
                print(f"✅ {user_name} נשמר באקסל!")
            except Exception as e:
                print(f"שגיאה באקסל: {e}")

            await update.message.reply_text("🔍 סורק דירות עבורך ב-Realta...")

            result_msg = search_and_recommend(
                city=info.get("area", "תל אביב"),
                rooms=info.get("rooms", "3"),
                budget=info.get("budget", "5000"),
                preferences={
                    "חניה": info.get("parking", ""),
                    "מרוהטת": info.get("furnished", ""),
                    "ממד": info.get("mamad", ""),
                    "קומה": info.get("floor", "")
                }
            )

            await update.message.reply_text(result_msg)

        except Exception as e:
            print(f"שגיאה: {e}")
            await update.message.reply_text("מחפש עבורך, אחזור עם תוצאות בקרוב!")

    else:
        await update.message.reply_text(reply)

        # שלב 2 — ספירת שאלות מכוונות
        if searching[user_id]:
            questions_asked[user_id] += 1
            print(f"שאלות שנשאלו: {questions_asked[user_id]}")

            # שלב 3 — אחרי 4 שאלות שלח המלצה מעודכנת
            if questions_asked[user_id] >= 4:
                questions_asked[user_id] = 0
                searching[user_id] = False

                try:
                    info = extract_client_info(conversations[user_id])

                    await update.message.reply_text("🎯 מעדכן את החיפוש לפי הפרטים שלך...")

                    result_msg = search_and_recommend(
                        city=info.get("area", "תל אביב"),
                        rooms=info.get("rooms", "3"),
                        budget=info.get("budget", "5000"),
                        preferences={
                            "חניה": info.get("parking", ""),
                            "מרוהטת": info.get("furnished", ""),
                            "ממד": info.get("mamad", ""),
                            "קומה": info.get("floor", "")
                        }
                    )

                    await update.message.reply_text(result_msg)
                    print(f"✅ המלצה מעודכנת נשלחה ל-{user_name}")

                except Exception as e:
                    print(f"שגיאה בהמלצה מעודכנת: {e}")

    print(f"{user_name}: {user_message}")
    print(f"רועי: {reply}")

def main():
    token = os.getenv("TELEGRAM_TOKEN")
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("רועי פעיל בטלגרם! שלח לו הודעה ב-t.me/Roinadlanbot")
    app.run_polling()

if __name__ == "__main__":
    main()
