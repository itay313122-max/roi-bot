FROM python:3.11-slim

# Create secure user for Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Copy all files into container
COPY --chown=user . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Expose port for health check (Hugging Face requirement)
EXPOSE 7860

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Run the bot
CMD ["python", "app.py"]