import asyncio
import logging
import os
import sys
import signal
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI
    import uvicorn
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Run: pip install fastapi uvicorn[standard]")
    sys.exit(1)

try:
    from telegram.ext import Application
except ImportError as e:
    print(f"❌ Failed to import telegram: {e}")
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# Global telegram app
telegram_app = None
telegram_app_running = False


def setup_telegram_bot_handlers(app: Application):
    """Register all telegram bot handlers."""
    from telegram.ext import CallbackQueryHandler, MessageHandler, filters
    from telegram_bot import handle_pre_search_callback, handle_message
    
    logger.info("📱 Registering Telegram bot handlers...")
    # Register handlers
    app.add_handler(CallbackQueryHandler(handle_pre_search_callback, pattern=r"^pre_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("✅ Telegram bot handlers registered successfully")
    print("✅ Telegram bot handlers registered successfully")


async def run_telegram_bot():
    """Run the telegram bot application."""
    global telegram_app, telegram_app_running
    try:
        token = os.getenv("TELEGRAM_TOKEN")
        if not token:
            logger.error("❌ TELEGRAM_TOKEN not set in environment!")
            print("❌ TELEGRAM_TOKEN not set in environment!")
            return
        
        logger.info("🚀 Starting Telegram bot application...")
        print("🚀 Starting Telegram bot application...")
        
        # Create the Application
        telegram_app = Application.builder().token(token).build()
        
        # Setup handlers
        setup_telegram_bot_handlers(telegram_app)
        
        # Start the bot
        telegram_app_running = True
        logger.info("📱 Telegram bot polling started...")
        print("📱 Telegram bot polling started...")
        await telegram_app.run_polling(allowed_updates=["message", "callback_query", "my_chat_member"])
        
    except Exception as e:
        logger.error(f"❌ Telegram bot crashed: {e}", exc_info=True)
        print(f"❌ Telegram bot crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("🛑 Telegram bot polling stopped")
        telegram_app_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI and Telegram bot lifecycle."""
    # Startup
    logger.info("🚀 Initializing application...")
    print("🚀 Initializing application...")
    
    # Start telegram bot as a background task
    bot_task = asyncio.create_task(run_telegram_bot())
    
    # Give bot time to initialize
    await asyncio.sleep(2)
    
    if telegram_app_running:
        logger.info("✅ Application fully initialized")
        print("✅ Application fully initialized")
    else:
        logger.warning("⚠️ Telegram bot may not have started correctly")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")
    bot_task.cancel()
    try:
        await bot_task
    except asyncio.CancelledError:
        logger.info("🛑 Telegram bot task cancelled")


# Create FastAPI app
app = FastAPI(
    title="Real Estate Agent Bot",
    description="Health check for Telegram bot running on Hugging Face",
    lifespan=lifespan
)


@app.get("/")
async def health_check():
    """Health check endpoint for Hugging Face."""
    return {
        "status": "ok",
        "message": "🤖 Real Estate Bot is running!",
        "service": "Telegram Bot with Health Check",
        "bot_status": "active" if telegram_app_running else "initializing"
    }


@app.get("/health")
async def health():
    """Alternative health check endpoint."""
    return {
        "status": "alive",
        "bot": "active" if telegram_app_running else "inactive"
    }


if __name__ == "__main__":
    try:
        # Get port from environment or default to 7860 (Hugging Face default)
        port_str = os.getenv("PORT", "7860")
        try:
            port = int(port_str)
        except ValueError:
            print(f"❌ Invalid PORT value: {port_str}. Using default 7860.")
            port = 7860
        
        # CRITICAL: Print before server starts so Hugging Face sees it
        print(f"=" * 70)
        print(f"🌟 Health check server is listening on port {port}")
        print(f"🌟 Binding to 0.0.0.0:{port}")
        print(f"🌟 Telegram bot will run alongside FastAPI")
        print(f"=" * 70)
        logger.info(f"🌟 Starting FastAPI server on 0.0.0.0:{port}...")
        logger.info("📱 Telegram bot will run in async context")
        
        # Run FastAPI with uvicorn on 0.0.0.0:7860
        print(f"\n📢 Starting server...\n")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,
            use_colors=False
        )
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}", exc_info=True)
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)