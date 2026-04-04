import asyncio
import re
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters
import os
from dotenv import load_dotenv
from agent_brain import think, extract_client_info
from excel_reporter import add_client
from yad2_scraper import (
    google_maps_url,
    search_and_recommend,
    deep_analyze_property_full,
    generate_facebook_search_url,
    scrape_rss,
    parse_budget_int,
    deep_analyze_property,
    _format_advanced_property_message,
)

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
        print("❌ No pending search found!")
        return
    
    info = pending["info"]
    user_name = pending["user_name"]

    searching[user_id] = True
    questions_asked[user_id] = 0

    print(f"\n{'='*70}")
    print(f"🔥 RUN_PENDING_SEARCH TRIGGERED FOR: {user_name}")
    print(f"{'='*70}")
    print(f"📋 Info extracted: {info}")
    print(f"📋 Listing type: {context.user_data.get('listing_type')}")
    print(f"📋 Buyer profile: {context.user_data.get('buyer_profile')}")

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
            print(f"⚠️  שגיאה באקסל: {e}")

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="🔍 סורק דירות עבורך ברשת המודעות וקבוצות פייסבוק...",
        )

        print(f"\n📞 Calling search_and_recommend with:")
        city = info.get("area", "תל אביב")
        rooms = info.get("rooms", "3")
        budget = info.get("budget", "5000")
        listing_type = context.user_data.get("listing_type")
        buyer_profile = context.user_data.get("buyer_profile", "קונה/שוכר")
        
        print(f"   • city: {city}")
        print(f"   • rooms: {rooms}")
        print(f"   • budget: {budget}")
        print(f"   • listing_type: {listing_type}")
        print(f"   • buyer_profile: {buyer_profile}")

        await search_and_recommend(
            update,
            context,
            city=city,
            rooms=rooms,
            budget=budget,
            preferences={
                "חניה": info.get("parking", ""),
                "מרוהטת": info.get("furnished", ""),
                "ממד": info.get("mamad", ""),
                "קומה": info.get("floor", ""),
            },
            listing_type=listing_type,
            buyer_profile=buyer_profile,
            user_name=user_name,
        )
        print(f"\n✅ search_and_recommend completed for {user_name}")
    except Exception as e:
        print(f"❌ Error in run_pending_search: {e}")
        import traceback
        traceback.print_exc()
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
    """Handle incoming messages from users."""
    try:
        user_id = update.effective_user.id
        user_name = update.effective_user.first_name
        user_message = update.message.text

        if not user_message:
            return

        # Handle restart command
        if user_message.strip().lower() in ("התחל מחדש", "restart", "/restart", "מחדש"):
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            conversations[user_id] = []
            searching[user_id] = False
            questions_asked[user_id] = 0
            print(f"🔄 {user_name} התחיל מחדש את השיחה")
            try:
                from config import FIRST_MESSAGE
                await update.message.reply_text(FIRST_MESSAGE)
            except Exception as e:
                print(f"❌ Error sending FIRST_MESSAGE: {e}")
                await update.message.reply_text("שלום! אני רועי. מחפשים דירה? בואנו נמצא לך אחת טובה!")
            return

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
            except Exception as inner_e:
                print(f"שגיאה ברישום לקוח חדש לאקסל: {inner_e}")

        # Check for direct scraper trigger (if user sends city/rooms/budget directly)
        if any(keyword in user_message for keyword in ["דירות", "חפש", "סורק", "apartments", "search"]):
            print(f"\n🔍 Direct search keyword detected: '{user_message}'")
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
                # Try to extract info from user message directly
                info = extract_client_info(conversations[user_id] + [{"role": "user", "content": user_message}])
                
                if info.get("area") and info.get("budget"):
                    print(f"✅ Direct search triggered with info: {info}")
                    
                    # Send aggressive confirmation
                    await update.message.reply_text(
                        f"🔥 בדיוק! {info.get('rooms', '3')} חדרים ב{info.get('area')} עד {info.get('budget')} שקל. "
                        f"מצאתי לך כמה נכסים שיחטפו מהר, תסתכל..."
                    )
                    
                    # Call scraper directly
                    city = info.get("area", "תל אביב")
                    rooms = info.get("rooms", "3")
                    budget_int = parse_budget_int(str(info.get("budget", "5000")), "השכרה")
                    
                    print(f"📡 Calling scrape_rss directly with: city={city}, rooms={rooms}, budget={budget_int}")
                    properties = await asyncio.to_thread(scrape_rss, city, rooms, budget_int, None)
                    
                    if properties:
                        print(f"✅ Found {len(properties)} properties!")
                        
                        # Display top 3 properties with advanced analytical format
                        for idx, prop in enumerate(properties[:3], 1):
                            try:
                                # Analyze property
                                analysis, scores = await asyncio.to_thread(
                                    deep_analyze_property,
                                    prop["url"],
                                    prop["title"],
                                    "קונה/שוכר",
                                )
                                
                                # Format with advanced analytical structure
                                advanced_msg = _format_advanced_property_message(
                                    num=idx,
                                    title=prop["title"],
                                    city=city,
                                    price=prop["price"],
                                    url=prop["url"],
                                    analysis_raw=analysis,
                                    scores=scores,
                                    address=prop.get("title", "")[:100],
                                    fresh=False,
                                    publisher="מקור",
                                    fee_line="",
                                )
                                
                                await update.message.reply_text(advanced_msg)
                            except Exception as prop_e:
                                print(f"⚠️  Failed to format property {idx}: {prop_e}")
                                # Fallback to simple display
                                fallback = f"🏠 דירה {idx}\n• {prop['title'][:50]}\n• 💰 {prop['price']:,} ₪\n• 🔗 {prop['url']}"
                                await update.message.reply_text(fallback)
                        
                        # Summary and tips
                        summary = f"\n✅ **מצאתי עבורך {len(properties[:3])} מציאות חמות מעל הראש!**"
                        await update.message.reply_text(summary)
                    else:
                        await update.message.reply_text(f"❌ אין יד2 בתחום הזה עכשיו. נסה שוב עם תקציב אחר או עיר שונה.")
                    
                    # Add to conversation
                    conversations[user_id].append({"role": "user", "content": user_message})
                    conversations[user_id].append({"role": "assistant", "content": f"תוצאות חיפוש: {len(properties)} דירות"})
                    return
            except Exception as search_e:
                print(f"⚠️  Direct scraper trigger failed: {search_e}")

        if context.user_data.get("awaiting_property_choice"):
            low = user_message.strip().lower()
            if low in ("ביטול", "לא", "לא עכשיו", "תודה"):
                context.user_data["awaiting_property_choice"] = False
            else:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
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

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        reply = think(conversations[user_id])

        conversations[user_id].append({
            "role": "assistant",
            "content": reply
        })

        # Check if bot is ready to start searching
        search_trigger_phrases = ["מתחיל לחפש", "אחפש עבורך", "מעולה"]
        should_trigger_search = any(phrase in reply for phrase in search_trigger_phrases) and not searching[user_id]
        
        if should_trigger_search:
            print(f"🔍 Search trigger detected in reply: {reply[:100]}")
            await update.message.reply_text(reply)
            try:
                info = extract_client_info(conversations[user_id])
                print(f"📋 Extracted client info: {info}")
                
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
                print(f"❌ שגיאה בהכנת חיפוש: {e}")
                import traceback
                traceback.print_exc()
                await update.message.reply_text("משהו השתבש בהכנת החיפוש — נסה שוב.")

            print(f"{user_name}: {user_message}")
            print(f"רועי: {reply}")
            return

        else:
            await update.message.reply_text(reply)

            if searching[user_id]:
                questions_asked[user_id] += 1
                print(f"📊 שאלות שנשאלו: {questions_asked[user_id]}/4")

                if questions_asked[user_id] >= 4:
                    questions_asked[user_id] = 0
                    searching[user_id] = False

                    try:
                        info = extract_client_info(conversations[user_id])
                        print(f"🔄 Updating search with refined criteria: {info}")

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
                        print(f"❌ שגיאה בהמלצה מעודכנת: {e}")
                        import traceback
                        traceback.print_exc()

        print(f"{user_name}: {user_message}")
        print(f"רועי: {reply}")

    except Exception as e:
        print(f"❌ Unexpected error in handle_message: {e}")
        import traceback
        traceback.print_exc()
        try:
            await update.message.reply_text("מצטער, משהו השתבש. נסה שוב בבקשה.")
        except:
            pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    try:
        from config import FIRST_MESSAGE
        await update.message.reply_text(FIRST_MESSAGE)
    except Exception as e:
        print(f"❌ Error in start handler: {e}")
        await update.message.reply_text("שלום! אני רועי. מחפשים דירה? בואנו נמצא לך אחת טובה!")


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks from inline keyboards."""
    await handle_pre_search_callback(update, context)


def main():
    """Main function to start the bot."""
    token = os.getenv("TELEGRAM_TOKEN")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 רועי פעיל בטלגרם!")
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
