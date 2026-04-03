import asyncio
import re
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import os
from dotenv import load_dotenv
from agent_brain import think, extract_client_info
from excel_reporter import add_client
from yad2_scraper import google_maps_url, search_and_recommend, deep_analyze_property_full, generate_facebook_search_url

load_dotenv()

conversations = {}
searching = {}
questions_asked = {}


def _parse_apartment_choice(text: str) -> Optional[int]:
    t = (text or "").strip()
    if t in ("1", "2", "3"):
        return int(t)
    m = re.match(r"דירה\s*([123])", t)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b([123])\b", t)
    if m2 and len(t) <= 12:
        return int(m2.group(1))
    return None


async def run_pending_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    pending = context.user_data.pop("pending_search", None)
    if not pending:
        return
    info = pending["info"]
    user_name = pending["user_name"]

    searching[user_id] = True
    questions_asked[user_id] = 0

    try:
        try:
            add_client(
                name=user_name,
                area=info.get("area", "לא ידוע"),
                rooms=info.get("rooms", "לא ידוע"),
                budget=info.get("budget", "לא ידוע"),
                deal_type=context.user_data.get("listing_type", info.get("deal_type", "לא ידוע")),
                notes=info.get("notes", ""),
            )
            print(f"✅ {user_name} נשמר באקסל!")
        except Exception as e:
            print(f"שגיאה באקסל: {e}")

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="🔍 סורק דירות עבורך ברשת המודעות וקבוצות פייסבוק...",
        )

        await search_and_recommend(
            update,
            context,
            city=info.get("area", "תל אביב"),
            rooms=info.get("rooms", "3"),
            budget=info.get("budget", "5000"),
            preferences={
                "חניה": info.get("parking", ""),
                "מרוהטת": info.get("furnished", ""),
                "ממד": info.get("mamad", ""),
                "קומה": info.get("floor", ""),
            },
            listing_type=context.user_data.get("listing_type"),
            buyer_profile=context.user_data.get("buyer_profile", "קונה/שוכר"),
            user_name=user_name,
        )
    except Exception as e:
        print(f"שגיאה: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="מחפש עבורך, אחזור עם תוצאות בקרוב!",
        )


async def handle_pre_search_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""

    if data in ("pre_ls_m", "pre_ls_r"):
        context.user_data["listing_type"] = "מכירה" if data == "pre_ls_m" else "השכרה"
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("קונה/שוכר", callback_data="pre_bp_res"),
                InlineKeyboardButton("משקיע", callback_data="pre_bp_inv"),
            ],
        ])
        await query.edit_message_text(
            text="האם אתה קונה/שוכר או משקיע?",
            reply_markup=kb,
        )
        return

    if data in ("pre_bp_res", "pre_bp_inv"):
        context.user_data["buyer_profile"] = "משקיע" if data == "pre_bp_inv" else "קונה/שוכר"
        await query.edit_message_text(text="מעולה! מתחיל לחפש עבורך...")
        await run_pending_search(update, context)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    user_message = update.message.text

    if user_id not in conversations:
        conversations[user_id] = []
        searching[user_id] = False
        questions_asked[user_id] = 0
        print(f"לקוח חדש: {user_name}")
        try:
            add_client(
                name=user_name,
                area="לא ידוע",
                rooms="לא ידוע",
                budget="לא ידוע",
                deal_type="לא ידוע",
                notes="נכנס לראשונה",
            )
        except Exception as e:
            print(f"שגיאה ברישום לקוח חדש לאקסל: {e}")

    if context.user_data.get("awaiting_property_choice"):
        low = user_message.strip().lower()
        if low in ("ביטול", "לא", "לא עכשיו", "תודה"):
            context.user_data["awaiting_property_choice"] = False
        else:
            pick = _parse_apartment_choice(user_message)
            props = context.user_data.get("last_properties") or []
            if pick is not None and 1 <= pick <= len(props):
                chosen = props[pick - 1]
                await update.message.reply_text("מכין דוח מעמיק...")
                report = await asyncio.to_thread(
                    deep_analyze_property_full,
                    chosen["url"],
                    chosen["title"],
                    chosen.get("buyer_profile", "קונה/שוכר"),
                )
                addr = chosen.get("address") or chosen.get("title") or ""
                nav = InlineKeyboardMarkup([
                    [InlineKeyboardButton("📍 נווט לדירה", url=google_maps_url(addr))],
                ])
                await update.message.reply_text(report, reply_markup=nav)
                context.user_data["awaiting_property_choice"] = False
                conversations[user_id].append({
                    "role": "user",
                    "content": user_message,
                })
                conversations[user_id].append({
                    "role": "assistant",
                    "content": report,
                })
                print(f"{user_name}: {user_message}")
                print(f"רועי: דוח מלא לדירה {pick}")
                return
            await update.message.reply_text(
                "כדי לבחור דירה לניתוח מעמיק כתוב 1, 2 או 3 (או \"דירה 1\"). "
                "או כתוב ביטול כדי להמשיך בשיחה."
            )
            return

    conversations[user_id].append({
        "role": "user",
        "content": user_message
    })

    reply = think(conversations[user_id])

    conversations[user_id].append({
        "role": "assistant",
        "content": reply
    })

    if ("מתחיל לחפש" in reply or "אחפש עבורך" in reply) and not searching[user_id]:
        await update.message.reply_text(reply)
        try:
            info = extract_client_info(conversations[user_id])
            context.user_data.pop("listing_type", None)
            context.user_data.pop("buyer_profile", None)
            context.user_data["pending_search"] = {"info": info, "user_name": user_name}
            kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("מכירה", callback_data="pre_ls_m"),
                    InlineKeyboardButton("השכרה", callback_data="pre_ls_r"),
                ],
            ])
            await update.message.reply_text(
                "האם מחפשים למכירה או להשכרה?",
                reply_markup=kb,
            )
        except Exception as e:
            print(f"שגיאה בהכנת חיפוש: {e}")
            await update.message.reply_text("משהו השתבש בהכנת החיפוש — נסה שוב.")

        print(f"{user_name}: {user_message}")
        print(f"רועי: {reply}")
        return

    else:
        await update.message.reply_text(reply)

        if searching[user_id]:
            questions_asked[user_id] += 1
            print(f"שאלות שנשאלו: {questions_asked[user_id]}")

            if questions_asked[user_id] >= 4:
                questions_asked[user_id] = 0
                searching[user_id] = False

                try:
                    info = extract_client_info(conversations[user_id])

                    await update.message.reply_text("🎯 מעדכן את החיפוש לפי הפרטים שלך...")

                    await search_and_recommend(
                        update,
                        context,
                        city=info.get("area", "תל אביב"),
                        rooms=info.get("rooms", "3"),
                        budget=info.get("budget", "5000"),
                        preferences={
                            "חניה": info.get("parking", ""),
                            "מרוהטת": info.get("furnished", ""),
                            "ממד": info.get("mamad", ""),
                            "קומה": info.get("floor", ""),
                        },
                        listing_type=context.user_data.get("listing_type"),
                        buyer_profile=context.user_data.get("buyer_profile", "קונה/שוכר"),
                        user_name=user_name,
                    )

                    print(f"✅ המלצה מעודכנת נשלחה ל-{user_name}")

                except Exception as e:
                    print(f"שגיאה בהמלצה מעודכנת: {e}")

    print(f"{user_name}: {user_message}")
    print(f"רועי: {reply}")


def main():
    token = os.getenv("TELEGRAM_TOKEN")
    app = Application.builder().token(token).build()
    app.add_handler(CallbackQueryHandler(handle_pre_search_callback, pattern=r"^pre_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("רועי פעיל בטלגרם! שלח לו הודעה ב-t.me/Roinadlanbot")
    app.run_polling()

if __name__ == "__main__":
    main()
