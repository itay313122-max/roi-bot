---
title: Roi Real Estate Bot
emoji: 🏠
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Roi - Real Estate Agent Bot 🏠

A Telegram bot that helps users find apartments in Israel using intelligent search and analysis. Deployed on HuggingFace Spaces with health check endpoint.

## Features

- **Smart Conversation**: Natural language understanding for apartment requirements
- **Fast Search**: Integration with Yad2 RSS feeds for real-time property listings
- **Deep Property Analysis**: AI-powered property evaluation using Groq API
- **Excel Reporting**: Client information tracking and management
- **Image Analysis**: Property image evaluation and scoring
- **Health Check**: Built-in web server for HuggingFace monitoring

## Environment Variables

Set these in HuggingFace Spaces Secrets:

- `TELEGRAM_TOKEN`: Your Telegram bot token (required)
- `GROQ_API_KEY`: API key for Groq (Llama 3 access) (required)

## How to Deploy on HuggingFace Spaces

1. Fork this repository
2. Create a new Space on HuggingFace (Docker SDK)
3. Connect your repository
4. Add environment variables in Secrets:
   - `TELEGRAM_TOKEN`
   - `GROQ_API_KEY`
5. The Space will automatically build and deploy
6. Bot will start on deployment

## Commands

- `/start` - Start the bot
- `/restart` or `התחל מחדש` - Start a new conversation

## Web Server

The bot runs a health check HTTP server on port 7860 (required for HuggingFace Spaces):
- Endpoint: `GET /` returns "Roi Bot is running!"
- This allows HuggingFace to monitor the Space status

## Tech Stack

- **python-telegram-bot**: Telegram Bot API wrapper
- **Groq**: LLM for conversation and analysis
- **BeautifulSoup4**: Web scraping for property data
- **Python threading**: Concurrent bot and web server execution
