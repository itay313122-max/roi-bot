FROM python:3.10-slim

# יצירת משתמש מאובטח עבור Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# העתקת כל הקבצים לתוך השרת
COPY --chown=user . .

# התקנת הספריות
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Expose port for health check (Hugging Face requirement)
EXPOSE 7860

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# הרצה של הקובץ הראשי (main.py) עם FastAPI + health check
CMD ["python", "main.py"]