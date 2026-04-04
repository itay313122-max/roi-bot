---
title: Roi Real Estate Bot
emoji: 🏠
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Roi - Real Estate Agent Bot 🏠

A Telegram bot that helps users find apartments in Israel using intelligent search and analysis.

## Features

- **Smart Conversation**: Natural language understanding for apartment requirements
- **Fast Search**: Integration with Yad2 RSS feeds for real-time property listings
- **Deep Property Analysis**: AI-powered property evaluation using Groq API
- **Excel Reporting**: Client information tracking and management
- **Image Analysis**: Property image evaluation and scoring

## Environment Variables

Set these in HuggingFace Spaces Secrets:

- `TELEGRAM_TOKEN`: Your Telegram bot token
- `GROQ_API_KEY`: API key for Groq (Llama 3 access)

## How to Deploy

1. Fork this repository
2. Create a new Space on HuggingFace with Docker SDK
3. Add the environment variables to Secrets
4. Push the code to the Space
5. The bot will automatically start

## Commands

- `/restart` or `התחל מחדש` - Start a new conversation

## Tech Stack

- **python-telegram-bot**: Telegram Bot API wrapper
- **Groq**: LLM for conversation and analysis
- **BeautifulSoup4**: Web scraping for property data
- **FastAPI**: Optional API endpoints
