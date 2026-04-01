import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://realta.co.il/en/ramat-gan/"
}

# RSS feed של Realta — הם מפרסמים אחד
url = "https://realta.co.il/feed.xml"

response = requests.get(url, headers=HEADERS, timeout=15)
print("סטטוס:", response.status_code)
print(response.text[:2000])
