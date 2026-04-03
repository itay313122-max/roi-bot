import asyncio
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI
    import uvicorn
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Run: pip install fastapi uvicorn[standard]")
    sys.exit(1)

try:
    from telegram_bot import main as start_telegram_bot
except ImportError as e:
    print(f"❌ Failed to import telegram_bot: {e}")
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Store bot task for lifecycle management
bot_thread = None


def run_telegram_bot_safely():
    """Run Telegram bot with exception handling."""
    try:
        logger.info("📱 Telegram bot polling started")
        print("📱 Telegram bot polling started (print for visibility)")
        start_telegram_bot()
    except Exception as e:
        logger.error(f"❌ Telegram bot crashed: {e}", exc_info=True)
        print(f"❌ Telegram bot crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("🛑 Telegram bot polling ended")
        print("🛑 Telegram bot polling ended")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage bot startup and shutdown with FastAPI lifecycle."""
    # Startup - Start bot in background thread BEFORE uvicorn is fully ready
    logger.info("🚀 Initializing Telegram bot in background thread...")
    global bot_thread
    
    try:
        # Create and start bot thread as daemon
        bot_thread = threading.Thread(target=run_telegram_bot_safely, daemon=True, name="TelegramBot")
        bot_thread.start()
        logger.info("✅ Telegram bot thread started successfully")
        print("✅ Telegram bot thread started successfully (print for visibility)")
        
        # Give bot a moment to initialize
        time.sleep(1)
    except Exception as e:
        logger.error(f"❌ Failed to start Telegram bot thread: {e}", exc_info=True)
        print(f"❌ Failed to start Telegram bot thread: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 FastAPI server shutting down, Telegram bot will terminate...")


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
        "service": "Telegram Bot with Health Check"
    }


@app.get("/health")
async def health():
    """Alternative health check endpoint."""
    is_bot_alive = bot_thread and bot_thread.is_alive()
    return {
        "status": "alive",
        "bot": "active" if is_bot_alive else "inactive"
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
        print(f"=" * 60)
        print(f"🌟 Health check server is listening on port {port}")
        print(f"🌟 Binding to 0.0.0.0:{port}")
        print(f"=" * 60)
        logger.info(f"🌟 Starting FastAPI server on 0.0.0.0:{port}...")
        logger.info("📱 Telegram bot will run in background thread")
        
        # Run FastAPI with uvicorn on 0.0.0.0:7860
        print(f"\n📢 Starting uvicorn server...")
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