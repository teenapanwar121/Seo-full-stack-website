from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse
from nltk.corpus import wordnet as wn
import language_tool_python
from googletrans import Translator
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
import uuid
import os
from fastapi.staticfiles import StaticFiles
import socket
import whois
import http.client
import json  # for proper payload formatting
import ssl
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
from urllib.parse import quote_plus
import random
import string
import yt_dlp
import sqlite3
import json
import morse_talk as mt


# For paraphrasing and summarization
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer
)
import torch

# =========================
# APP INITIALIZATION
# =========================
app = FastAPI(title="SEO Tools API")

with open("std_codes.json", "r", encoding="utf-8") as f:
    STD_DATA = json.load(f)


# =========================
# CORS CONFIGURATION
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

os.makedirs("download", exist_ok=True)

# 2️⃣ Mount downloads folder for HTTP access
#app.mount("/download", StaticFiles(directory="download"), name="download")


# =========================
# INITIALIZE TOOLS
# =========================
tool = language_tool_python.LanguageTool('en-US')
translator = Translator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# PARAPHRASING MODEL (T5)
# =========================
paraphrase_model_name = "ramsrigouthamg/t5_paraphraser"
paraphrase_tokenizer = T5Tokenizer.from_pretrained(paraphrase_model_name)
paraphrase_model = T5ForConditionalGeneration.from_pretrained(
    paraphrase_model_name
).to(device)

# =========================
# SUMMARIZATION MODEL (BART)
# =========================
summarizer_model_name = "facebook/bart-large-cnn"
summarizer_tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = BartForConditionalGeneration.from_pretrained(
    summarizer_model_name
).to(device)

# =========================
# OFFLINE PROFANITY FILTER
# =========================
BAD_WORDS = {"Hell", "shit", "damn", "stupid", "idiot", "deaf","hell","Shit","Dam","Stupid","Bastrad","Bitch","bastrad"}

def contains_profanity(text: str) -> bool:
    return any(word in BAD_WORDS for word in text.lower().split())

def censor_text(text: str) -> str:
    return " ".join(
        "*" * len(word) if word.lower() in BAD_WORDS else word
        for word in text.split()
    )

# =========================
# REQUEST MODELS
# =========================
class SpellRequest(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    text: str
    target_lang: str

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

class ParaphraseRequest(BaseModel):
    text: str
    num_return_sequences: int = 1

class WordCountRequest(BaseModel):
    text: str

class SummarizeRequest(BaseModel):
    text: str

class ProfanityRequest(BaseModel):
    text: str

class KeywordResearchRequest(BaseModel):
    keyword: str
class SeoAnalysisRequest(BaseModel):
    url: str
    lang: str = "en"

class SSLRequest(BaseModel):
    domain: str

class WebsiteContactRequest(BaseModel):
    url: str

class AgeCalculatorRequest(BaseModel):
    date_of_birth: str  # format: YYYY-MM-DD

class PasswordRequest(BaseModel):
    password: str

class BMIRequest(BaseModel):
    weight_kg: float
    height_cm: float

class ReverseGeoRequest(BaseModel):
    latitude: float
    longitude: float

class UrlShortenRequest(BaseModel):
    long_url: str

class QRCodeRequest(BaseModel):
    content: str      # URL / text to convert into QR
    size: int = 300   # QR size (default 300x300)

class TempMailRequest(BaseModel):
    username: str = None  # optional, random if not provided
    domain: str = "1secmail.com"  # default domain

class TempMailInboxRequest(BaseModel):
    email: str  # full email e.g., abc123@1secmail.com


class YouTubeMP3Request(BaseModel):
    url: str
    quality: str = "192"  # Default quality: 192kbps

class PincodeLookupRequest(BaseModel):
    pincode: str

class StdCodeRequest(BaseModel):
    state: str
    city: str

class MorseCodeRequest(BaseModel):
    text: str

class SocialLinksRequest(BaseModel):
    url: str


# ✅ PLAGIARISM REQUEST MODEL
class PlagiarismRequest(BaseModel):
    text: str

class SimilarWordsRequest(BaseModel):
    word: str    

# =========================
# SPELL CHECK API
# =========================
@app.post("/check")
def check_spelling(data: SpellRequest):
    matches = tool.check(data.text)
    return {
        "corrections": [
            {
                "wrong_word": data.text[m.offset:m.offset + m.error_length],
                "suggestion": m.replacements[0] if m.replacements else "No suggestion"
            }
            for m in matches
        ]
    }
# =========================
# TEMP MAIL HELPER FUNCTIONS
# =========================
def random_username(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


# =========================
# TRANSLATION API
# =========================
@app.post("/translate")
def translate_text(data: TranslateRequest):
    result = translator.translate(data.text, dest=data.target_lang)
    return {
        "original_text": data.text,
        "translated_text": result.text,
        "target_language": data.target_lang
    }

@app.post("/tts")
def text_to_speech(data: TTSRequest):
    try:
        # 1️⃣ Create audio folder if not exists
        os.makedirs("audio", exist_ok=True)

        # 2️⃣ Generate unique filename
        filename = f"audio/{uuid.uuid4()}.mp3"

        # 3️⃣ Generate speech
        tts = gTTS(text=data.text, lang=data.language)
        tts.save(filename)

        # 4️⃣ Return audio file
        return FileResponse(
            path=filename,
            media_type="audio/mpeg",
            filename="speech.mp3"
        )

    except Exception as e:
        return {"error": str(e)}

# =========================
# PARAPHRASING API (PRODUCTION SAFE)
# =========================
@app.post("/paraphrase")
def paraphrase_text(data: ParaphraseRequest):

    # ---- Basic validation ----
    if not data.text or len(data.text.strip()) == 0:
        return {
            "status": "failed",
            "error": "Input text cannot be empty"
        }

    if len(data.text) > 500:
        return {
            "status": "failed",
            "error": "Text too long. Maximum 500 characters allowed."
        }

    # ---- Model input ----
    input_text = f"paraphrase: {data.text} </s>"

    encoding = paraphrase_tokenizer.encode_plus(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="longest"
    )

    # ---- Generate ONE best paraphrase ----
    outputs = paraphrase_model.generate(
        encoding["input_ids"].to(device),
        attention_mask=encoding["attention_mask"].to(device),
        num_beams=5,
        num_return_sequences=1,   # 🔑 IMPORTANT
        max_length=512,
        temperature=1.0
    )

    paraphrased_text = paraphrase_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # ---- Clean, frontend-safe response ----
    return {
        "status": "success",
        "paraphrased_text": paraphrased_text
    }

# =========================
# WORD COUNT API
# =========================
@app.post("/wordcount")
def word_count(data: WordCountRequest):
    text = data.text.strip()
    return {
        "word_count": len(text.split()),
        "character_count": len(text)
    }

# =========================
# SUMMARIZATION API
# =========================
@app.post("/summarize")
def summarize_text(data: SummarizeRequest):
    inputs = summarizer_tokenizer.encode(
        data.text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)

    summary_ids = summarizer_model.generate(
        inputs,
        max_length=150,
        min_length=40,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    return {
        "original_text": data.text,
        "summary": summarizer_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
    }

# =========================
# PROFANITY FILTER API
# =========================
@app.post("/profanity")
def profanity_filter(data: ProfanityRequest):
    return {
        "original_text": data.text,
        "contains_profanity": contains_profanity(data.text),
        "censored_text": censor_text(data.text)
    }

# =========================
# KEYWORD RESEARCH API (SEO)
# =========================
@app.post("/keyword-research")
def keyword_research(data: KeywordResearchRequest):
    keyword = data.keyword.lower()

    related_keywords = [
        f"best {keyword}",
        f"{keyword} for beginners",
        f"cheap {keyword}",
        f"{keyword} in india",
        f"{keyword} reviews"
    ]

    return {
        "input_keyword": keyword,
        "keyword_type": "Long-tail keyword",
        "related_keywords": related_keywords,
        "usage": "Use these keywords in title, headings, and content for SEO"
    }
# =========================
# ✅ PLAGIARISM CHECK USING RAPIDAPI
# =========================
@app.post("/plagiarism-check")
def plagiarism_check(data: PlagiarismRequest):
    import http.client  # we can import here or at the top

    conn = http.client.HTTPSConnection("plagiarism-checker-and-auto-citation-generator-multi-lingual.p.rapidapi.com")

    payload = json.dumps({
        "text": data.text,
        "language": "en",
        "includeCitations": False,
        "scrapeSources": False
    })

    headers = {
        'x-rapidapi-key': "ea59f31bcamsh709e26b6a05976cp16e5a2jsn48ac4f87257c",
        'x-rapidapi-host': "plagiarism-checker-and-auto-citation-generator-multi-lingual.p.rapidapi.com",
        'Content-Type': "application/json"
    }

    conn.request("POST", "/plagiarism", payload, headers)

    res = conn.getresponse()
    response_data = res.read()

    return json.loads(response_data)


# =========================
# WEBSITE TRACKING – IP DETAILS
# =========================
@app.get("/track-ip")
def track_ip(request: Request, domain: str = None):
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        client_ip = x_forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else None

    try:
        hostname = socket.gethostname()
        server_local_ip = socket.gethostbyname(hostname)
    except Exception:
        server_local_ip = None

    website_ip = None
    if domain:
        try:
            website_ip = socket.gethostbyname(domain)
        except Exception:
            website_ip = None

    return {
        "client_ip": client_ip,
        "server_local_ip": server_local_ip,
        "website_ip": website_ip
    }

# =========================
# WEBSITE TRACKING – DOMAIN DETAILS
# =========================
@app.get("/track-domain")
def track_domain(domain: str):
    try:
        domain_ip = socket.gethostbyname(domain)
        domain_info = whois.whois(domain)

        return {
            "domain": domain,
            "domain_ip": domain_ip,
            "registrar": domain_info.registrar,
            "creation_date": str(domain_info.creation_date),
            "expiration_date": str(domain_info.expiration_date),
            "name_servers": domain_info.name_servers,
            "country": domain_info.country
        }

    except Exception as e:
        return {
            "domain": domain,
            "error": str(e)
        }
# =========================
# BULK DOMAIN DA-PA REQUEST MODEL
# =========================
class BulkDARequest(BaseModel):
    urls: list  # list of domains/URLs

# =========================
# BULK DOMAIN DA-PA CHECK API
# =========================
@app.post("/bulk-da-pa")
def bulk_da_pa_check(data: BulkDARequest):
    conn = http.client.HTTPSConnection("bulk-domain-da-pa-check.p.rapidapi.com")

    payload = json.dumps({"urls": data.urls})

    headers = {
        'x-rapidapi-key': "ea59f31bcamsh709e26b6a05976cp16e5a2jsn48ac4f87257c",  
        'x-rapidapi-host': "bulk-domain-da-pa-check.p.rapidapi.com",
        'Content-Type': "application/json"
    }

    conn.request("POST", "/check", payload, headers)
    res = conn.getresponse()
    response_data = res.read()

    return json.loads(response_data)
@app.post("/seo-page-analysis")
def seo_page_analysis(data: SeoAnalysisRequest):
    import http.client
    import json

    conn = http.client.HTTPSConnection(
        "seo-analysis-report-improve-website-ai-seo-tool.p.rapidapi.com"
    )

    payload = json.dumps({
        "url": data.url,
        "lang": data.lang
    })

    headers = {
        'x-rapidapi-key': "ea59f31bcamsh709e26b6a05976cp16e5a2jsn48ac4f87257c",  
        'x-rapidapi-host': "seo-analysis-report-improve-website-ai-seo-tool.p.rapidapi.com",
        'Content-Type': "application/json"
    }

    conn.request("POST", "/check?noqueue=1", payload, headers)

    res = conn.getresponse()
    response_data = res.read()

    return json.loads(response_data)

@app.post("/ssl-info")
def get_ssl_info(data: SSLRequest):
    context = ssl.create_default_context()

    try:
        with socket.create_connection((data.domain, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=data.domain) as ssock:
                cert = ssock.getpeercert()

        expiry_date = datetime.strptime(
            cert['notAfter'], "%b %d %H:%M:%S %Y %Z"
        )

        return {
            "domain": data.domain,
            "https_enabled": True,
            "issuer": cert.get("issuer"),
            "valid_from": cert.get("notBefore"),
            "valid_until": cert.get("notAfter"),
            "expired": expiry_date < datetime.utcnow()
        }

    except Exception as e:
        return {
            "domain": data.domain,
            "https_enabled": False,
            "error": str(e)
        }
@app.post("/website-contacts")
def website_contacts(data: WebsiteContactRequest):
    url = data.url.strip()

    if not url.startswith("http"):
        url = "https://" + url

    headers = {
        "User-Agent": "Mozilla/5.0 (SEO Analyzer Bot)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=6)
        soup = BeautifulSoup(response.text, "html.parser")

        text = soup.get_text(" ")

        # 📧 Email extraction
        emails = list(set(re.findall(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            text
        )))

        # 📞 Phone number extraction
        phones = list(set(re.findall(
            r"\+?\d[\d\s().-]{7,}\d",
            text
        )))

        # 🔗 Contact-related pages
        contact_pages = []
        for link in soup.find_all("a", href=True):
            href = link["href"].lower()
            if any(word in href for word in ["contact", "about", "support", "help"]):
                contact_pages.append(urljoin(url, link["href"]))

        return {
            "website": url,
            "emails_found": emails,
            "phone_numbers_found": phones,
            "contact_pages": list(set(contact_pages)),
            "status": "success"
        }

    except Exception as e:
        return {
            "website": url,
            "error": str(e),
            "status": "failed"
        }

def get_social_links(url: str):
    if not url.startswith("http"):
        url = "https://" + url

    headers = {
        "User-Agent": "Mozilla/5.0 (SEO Tools Social Bot)"
    }

    response = requests.get(url, headers=headers, timeout=8)
    soup = BeautifulSoup(response.text, "html.parser")

    social_domains = {
        "facebook": "facebook.com",
        "instagram": "instagram.com",
        "twitter": "twitter.com",
        "linkedin": "linkedin.com",
        "youtube": "youtube.com",
        "tiktok": "tiktok.com",
        "pinterest": "pinterest.com"
    }

    found_links = {key: [] for key in social_domains}

    for link in soup.find_all("a", href=True):
        href = link["href"]
        for platform, domain in social_domains.items():
            if domain in href:
                found_links[platform].append(href)

    # remove duplicates
    for platform in found_links:
        found_links[platform] = list(set(found_links[platform]))

    return found_links


@app.post("/social-links")
def social_links_finder(data: SocialLinksRequest):
    try:
        links = get_social_links(data.url)

        return {
            "status": "success",
            "website": data.url,
            "social_links": links
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


@app.post("/age-calculator")
def age_calculator(data: AgeCalculatorRequest):
    try:
        dob = datetime.strptime(data.date_of_birth, "%Y-%m-%d").date()
        today = datetime.utcnow().date()

        if dob > today:
            return {
                "status": "failed",
                "error": "Date of birth cannot be in the future"
            }

        years = today.year - dob.year
        months = today.month - dob.month
        days = today.day - dob.day

        if days < 0:
            months -= 1
            previous_month = (today.month - 1) or 12
            days_in_prev_month = (datetime(today.year, previous_month + 1, 1) - datetime(today.year, previous_month, 1)).days
            days += days_in_prev_month

        if months < 0:
            years -= 1
            months += 12

        return {
            "date_of_birth": data.date_of_birth,
            "age": {
                "years": years,
                "months": months,
                "days": days
            },
            "status": "success"
        }

    except ValueError:
        return {
            "status": "failed",
            "error": "Invalid date format. Use YYYY-MM-DD"
        }

def check_password_strength(password: str):
    strength_score = 0
    recommendations = []

    # Length
    if len(password) >= 8:
        strength_score += 1
    else:
        recommendations.append("Use at least 8 characters")

    # Lowercase
    if re.search(r"[a-z]", password):
        strength_score += 1
    else:
        recommendations.append("Add lowercase letters")

    # Uppercase
    if re.search(r"[A-Z]", password):
        strength_score += 1
    else:
        recommendations.append("Add uppercase letters")

    # Digits
    if re.search(r"\d", password):
        strength_score += 1
    else:
        recommendations.append("Add numbers")

    # Special characters
    if re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        strength_score += 1
    else:
        recommendations.append("Add special characters")

    # Strength description
    if strength_score <= 2:
        strength = "Weak"
    elif strength_score == 3 or strength_score == 4:
        strength = "Moderate"
    else:
        strength = "Strong"

    return {
        "strength_score": strength_score,
        "strength": strength,
        "recommendations": recommendations
    }

@app.post("/password-strength")
def password_strength(data: PasswordRequest):
    result = check_password_strength(data.password)
    return {
        "password": data.password,
        "strength_score": result["strength_score"],
        "strength": result["strength"],
        "recommendations": result["recommendations"]
    }

def calculate_bmi(weight_kg: float, height_cm: float):
    height_m = height_cm / 100

    if height_m <= 0 or weight_kg <= 0:
        return {
            "status": "failed",
            "error": "Height and weight must be positive values"
        }

    bmi = weight_kg / (height_m ** 2)
    bmi = round(bmi, 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return {
        "bmi": bmi,
        "category": category
    }

@app.post("/bmi-calculator")
def bmi_calculator(data: BMIRequest):
    result = calculate_bmi(data.weight_kg, data.height_cm)
    return {
        "weight_kg": data.weight_kg,
        "height_cm": data.height_cm,
        **result
    }

@app.post("/reverse-geocoding")
def reverse_geocoding(data: ReverseGeoRequest):
    lat = data.latitude
    lon = data.longitude

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2"
    }

    headers = {
        "User-Agent": "SEO-Tools-App/1.0 (admin@seotools.com)"
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=8
        )

        # 🚨 IMPORTANT: check response
        if response.status_code != 200:
            return {
                "status": "failed",
                "error": f"Nominatim error: {response.status_code}"
            }

        result = response.json()

        return {
            "latitude": lat,
            "longitude": lon,
            "location": result.get("display_name"),
            "address": result.get("address"),
            "status": "success"
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

@app.post("/url-shorten")
def shorten_url(data: UrlShortenRequest):
    long_url = data.long_url.strip()

    if not long_url.startswith("http"):
        long_url = "https://" + long_url

    providers = [
        "https://is.gd/create.php",
        "https://v.gd/create.php",
        "https://da.gd/s"
    ]

    headers = {
        "User-Agent": "SEO-Tools-URL-Shortener/1.0"
    }

    for provider in providers:
        try:
            if "da.gd" in provider:
                response = requests.get(
                    provider,
                    params={"url": long_url},
                    headers=headers,
                    timeout=6
                )

                if response.status_code == 200 and response.text.startswith("https"):
                    return {
                        "status": "success",
                        "provider": "da.gd",
                        "short_url": response.text.strip()
                    }

            else:
                response = requests.get(
                    provider,
                    params={
                        "format": "simple",
                        "url": long_url
                    },
                    headers=headers,
                    timeout=6
                )

                if response.status_code == 200 and response.text.startswith("http"):
                    return {
                        "status": "success",
                        "provider": provider.split("//")[1],
                        "short_url": response.text.strip()
                    }

        except Exception:
            continue

    return {
        "status": "failed",
        "error": "All URL shortener services are currently unavailable"
    }

@app.post("/generate-qr")
def generate_qr(data: QRCodeRequest):

    # 1️⃣ Validate input
    if not data.content.strip():
        return {
            "status": "failed",
            "error": "Content cannot be empty"
        }

    # 2️⃣ Encode content (important for URLs & spaces)
    encoded_content = quote_plus(data.content)

    # 3️⃣ Create QR image URL (FREE public API)
    qr_url = (
        f"https://api.qrserver.com/v1/create-qr-code/"
        f"?size={data.size}x{data.size}&data={encoded_content}"
    )

    # 4️⃣ Return response
    return {
        "status": "success",
        "input": data.content,
        "qr_url": qr_url,
        "message": "QR code generated successfully"
    }

# =========================
# CREATE TEMP EMAIL
# =========================
@app.post("/temp-mail/create")
def create_temp_mail(data: TempMailRequest):
    username = data.username or random_username()
    email = f"{username}@{data.domain}"
    return {
        "status": "success",
        "email": email,
        "message": "Temp email created. Use it to receive messages."
    }

@app.post("/temp-mail/inbox")
def get_inbox(data: TempMailInboxRequest):
    try:
        login, domain = data.email.split("@")
        api_url = f"https://www.1secmail.com/api/v1/?action=getMessages&login={login}&domain={domain}"
        resp = requests.get(api_url, timeout=6)

        # ✅ Check if response is valid JSON
        try:
            messages = resp.json()
        except ValueError:
            messages = []

        return {
            "status": "success",
            "email": data.email,
            "messages": messages  # will be empty list if no messages
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


# =========================
# READ EMAIL
# =========================
# =========================
# READ EMAIL
# =========================
@app.post("/temp-mail/read/{message_id}")
def read_email(data: TempMailInboxRequest, message_id: int):
    try:
        login, domain = data.email.split("@")
        api_url = f"https://www.1secmail.com/api/v1/?action=readMessage&login={login}&domain={domain}&id={message_id}"
        resp = requests.get(api_url, timeout=6)

        # ✅ Check if response is valid JSON
        try:
            message = resp.json()
        except ValueError:
            message = {}  # empty dict if no message or invalid response

        return {
            "status": "success",
            "email": data.email,
            "message": message  # empty if message not found
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
    
# =========================
# YOUTUBE MP3 DOWNLOADERs
# =========================
def download_youtube_mp3(url: str, quality: str = "192"):
    """
    Simple YouTube to MP3 downloader
    Returns: (file_path, title)
    """
    try:
        unique_id = str(uuid.uuid4())[:8]
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'download/%(title)s_{unique_id}.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': quality,
            }],
            'noplaylist': True,
            'ffmpeg_location': r"C:\ffmpeg\ffmpeg\bin\ffmpeg.exe",
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', f'audio_{unique_id}')
            
            # Find the downloaded file
            import glob
            mp3_files = glob.glob(f"download/*{unique_id}*.mp3")
            if mp3_files:
                return mp3_files[0], title
            else:
                return None, title
                
    except Exception as e:
        # Cleanup temp files
        import glob
        for temp_file in glob.glob("download/*.part"):
            try:
                os.remove(temp_file)
            except:
                pass
        raise e

@app.post("/youtube-to-mp3")
def youtube_to_mp3(data: YouTubeMP3Request):
    try:
        file_path, title = download_youtube_mp3(
            data.url,
            data.quality
        )

        if not file_path:
            return {
                "status": "failed",
                "error": "MP3 file not generated"
            }

        return FileResponse(
            path=file_path,
            media_type="audio/mpeg",
            filename=f"{title}.mp3"
        )

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


@app.post("/pincode-lookup")
def pincode_lookup(data: PincodeLookupRequest):
    pincode = data.pincode.strip()

    if not pincode.isdigit() or len(pincode) != 6:
        return {
            "status": "failed",
            "error": "Invalid pincode. Must be 6 digits."
        }

    url = f"http://api.postalpincode.in/pincode/{pincode}"

    headers = {
        "User-Agent": "Mozilla/5.0 (SEO Tools App)",
        "Accept": "application/json"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return {
                "status": "failed",
                "error": "India Post service unavailable"
            }

        result = response.json()

        if result[0]["Status"] != "Success":
            return {
                "status": "failed",
                "error": "No data found for this pincode"
            }

        post_offices = result[0]["PostOffice"]

        return {
            "status": "success",
            "pincode": pincode,
            "country": "India",
            "state": post_offices[0]["State"],
            "district": post_offices[0]["District"],
            "post_offices": [
                {
                    "name": po["Name"],
                    "branch_type": po["BranchType"],
                    "delivery_status": po["DeliveryStatus"]
                }
                for po in post_offices
            ]
        }

    except requests.exceptions.RequestException as e:
        return {
            "status": "failed",
            "error": "India Post API temporarily unavailable"
        }

@app.post("/std-code-lookup")
def std_code_lookup(data: StdCodeRequest):
    state = data.state.strip().lower()
    city = data.city.strip().lower()

    return {
        "status": "success",
        "std_code": STD_DATA[state][city]
    }

@app.post("/word-to-morse")
def word_to_morse(data: MorseCodeRequest):
    try:
        raw_morse = mt.encode(data.text)

        # Normalize spaces (library output → user friendly)
        morse_code = " ".join(raw_morse.split())

        return {
            "status": "success",
            "input_text": data.text,
            "morse_code": morse_code
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

@app.post("/similar-words")
def similar_words(data: SimilarWordsRequest):
    word = data.word.lower().strip()
    synonyms = set()

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))

    return {
        "word": word,
        "similar_words": list(synonyms)[:15]  # limit for SEO
    }
