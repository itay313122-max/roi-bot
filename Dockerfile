FROM python:3.10

# יצירת משתמש מאובטח עבור Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# העתקת כל הקבצים לתוך השרת
COPY --chown=user . .

# התקנת הספריות
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# הרצה של הקובץ הראשי (main.py)
CMD ["python", "main.py"]