# ---- Base image ----
    FROM python:3.10-slim

    # ---- Install Chrome for Selenium ----
    RUN apt-get update && \
        apt-get install -y chromium chromium-driver && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*
    
    # ---- Environment variables ----
    ENV CHROME_BIN=/usr/bin/chromium
    ENV PATH="/usr/lib/chromium:$PATH"
    
    # ---- Working directory ----
    WORKDIR /app
    
    # ---- Copy code ----
    COPY . /app
    
    # ---- Install dependencies ----
    RUN pip install --upgrade pip
    RUN pip install -r requirements.txt
    
    # ---- Streamlit port ----
    EXPOSE 8501
    
    # ---- Run app ----
    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    