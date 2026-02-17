import os
import json
import asyncio
import logging
import feedparser
import requests
import google.generativeai as genai
import edge_tts
from youtube_transcript_api import YouTubeTranscriptApi
from pathlib import Path
from dotenv import load_dotenv
import time
import random
import re

# --- 1. LOCAL SETUP ---
load_dotenv()

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

HISTORY_FILE = Path("history.json")

# --- LOAD FEEDS ---
try:
    with open("feeds.json", "r") as f:
        data = json.load(f)
        # Handle both list of strings and list of objects
        if data and isinstance(data[0], dict):
            YOUTUBE_FEEDS = [item["url"] for item in data if "url" in item]
        else:
            YOUTUBE_FEEDS = data
except Exception as e:
    logger.warning(f"⚠️ feeds.json error: {e}. Using empty list.")
    YOUTUBE_FEEDS = []

# --- FALLBACK GENERATOR (Master List) ---
def generate_with_fallback(prompt_parts):
    """
    Tries ALL known Gemini models in order of capability.
    Handles Rate Limits (429) and Not Found (404) errors gracefully.
    """
    # MASTER PRIORITY LIST
    models_to_try = [
        # --- TIER 1: The Smartest (Try these first) ---
        'gemini-2.5-flash'
        'gemini-2.0-flash-exp',       # Often smartest & fastest
        'gemini-1.5-pro',             # Best for complex reasoning
        'gemini-1.5-flash',           # Standard workhorse
        
        # --- TIER 2: The "Lite" & Fast (High Quota) ---
        'gemini-2.5-flash-lite',      # New efficient model
        'gemini-flash-lite-latest',   # Latest lite version
        'gemini-1.5-flash-8b',        # Extremely fast/cheap
        
        # --- TIER 3: Previews & Experimental (Often separate quotas) ---
        'gemini-2.5-flash-preview-09-2025',
        'gemini-2.5-flash-lite-preview-09-2025',
        
        # --- TIER 4: Open Models (Gemma - Good fallbacks) ---
        'gemma-3-27b-it',             # Smarter open model
        'gemma-3-9b-it',              # Mid-range
        'gemma-3-4b-it',              # Fast
        'gemma-3-1b-it'               # Ultra-fast/Low quality
    ]
    
    genai.configure(api_key=GEMINI_API_KEY)

    for model_name in models_to_try:
        try:
            # logger.info(f"🧠 Asking {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_parts)
            return response.text
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # 1. Handle Rate Limits (Busy)
            if "429" in error_msg or "quota" in error_msg:
                logger.warning(f"⚠️ {model_name} is Busy/Rate Limited. Switching to next...")
                time.sleep(1) # Short pause to be polite
                continue
            
            # 2. Handle 404 (Model doesn't exist/deprecated)
            elif "404" in error_msg or "not found" in error_msg:
                # logger.warning(f"⚠️ {model_name} not found. Skipping.")
                continue
                
            # 3. Handle Overloaded/Server Errors
            elif "503" in error_msg or "overloaded" in error_msg:
                logger.warning(f"⚠️ {model_name} Overloaded. Switching...")
                time.sleep(2)
                continue
                
            # 4. Other Errors (Safety, etc.)
            else:
                logger.warning(f"⚠️ {model_name} Error: {e}. Switching...")
                continue
    
    logger.error("❌ ALL models failed. Your API key is likely completely exhausted for the day.")
    return None

# --- HELPER FUNCTIONS ---
def load_history():
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text())
            return set(data.get("videos", []))
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
            return set()
    return set()

def save_history(history):
    try:
        history_list = list(history)
        if len(history_list) > 500: # Keep file size manageable
            history_list = history_list[-500:]
        
        HISTORY_FILE.write_text(
            json.dumps({"videos": history_list}, indent=2)
        )
    except Exception as e:
        logger.error(f"Failed to save history: {e}")

def get_transcript(video_id):
    logger.info(f"🕵️ Fetching transcript for {video_id}...")
    
    # --- METHOD 1: API (Fastest) ---
    try:
        # SMART CHANGE 1: Pass cookies to the Transcript API too!
        # This was silently failing before because it looked like a bot.
        if os.path.exists("cookies.txt"):
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, cookies="cookies.txt")
        else:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
        text = " ".join([entry['text'] for entry in transcript_list])
        return text
    except Exception as e:
        logger.warning(f"⚠️ API Method failed: {e}")
        pass # Silent fail to try next method

    # --- METHOD 2: yt-dlp (Robust Fallback for Captions) ---
    try:
        import yt_dlp
        time.sleep(random.uniform(2, 5)) 
        
        url = f"https://youtu.be/{video_id}"
        ydl_opts = {
            'skip_download': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'outtmpl': f'transcript_{video_id}',
            'quiet': True,
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'cookiefile': 'cookies.txt',
            # SMART CHANGE 2: Pretend to be an Android phone to bypass JS challenges
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Check for caption file
        found_text = None
        for file in os.listdir("."):
            if file.startswith(f"transcript_{video_id}") and file.endswith(".vtt"):
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                os.remove(file) # Cleanup
                
                # Cleaning VTT junk
                lines = content.splitlines()
                text = []
                for line in lines:
                    if "-->" not in line and "WEBVTT" not in line and line.strip():
                        clean_line = line.replace("<c>", "").replace("</c>", "").replace("&nbsp;", " ")
                        clean_line = re.sub(r'\[.*?\]', '', clean_line)
                        clean_line = re.sub(r'\(.*?\)', '', clean_line)
                        text.append(clean_line)
                found_text = " ".join(text)
                break
        
        if found_text:
            return found_text

    except Exception as e:
        logger.warning(f"⚠️ Method 2 failed: {e}")

    # --- METHOD 3: The Nuclear Option (Download Audio + Gemini Listen) ---
    try:
        import yt_dlp
        logger.info("☢️ Nuclear Option: Listening to audio...")
        
        filename = f"audio_{video_id}"
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '64',
            }],
            'outtmpl': filename,
            'quiet': True,
            'cookiefile': 'cookies.txt',
            # SMART CHANGE 2 (Again): Android spoofing for audio download
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://youtu.be/{video_id}"])
            
        final_audio_file = f"{filename}.mp3"
            
        if os.path.exists(final_audio_file):
            genai.configure(api_key=GEMINI_API_KEY)
            uploaded_file = genai.upload_file(final_audio_file)
            
            # Wait for processing
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = genai.get_file(uploaded_file.name)
            
            # USE FALLBACK GENERATOR HERE
            transcript_text = generate_with_fallback([
                "Listen to this audio and generate a full transcript of what is being said. Do not summarize, just transcribe.",
                uploaded_file
            ])
            
            # Cleanup
            try:
                os.remove(final_audio_file)
                uploaded_file.delete()
            except: pass
            
            return transcript_text
        else:
            return None
            
    except Exception as e:
        logger.error(f"❌ Nuclear Option failed: {e}")
        return None

def analyze_video(transcript, channel_name, video_title, video_url):
    
    prompt = f"""
    You are a high-energy, joyful Radio RJ who loves tech. You are talking to your listeners (friends).
    
    VIDEO: "{video_title}" by {channel_name}
    TRANSCRIPT: {transcript[:50000]}

    YOUR TASK:
    1. TELEGRAM POST (Short & Punchy):
       - 1 Hook Line (Make me curious).
       - 2 Short Bullet points on the "Why".
       - No filler. Just value.

    2. PODCAST SCRIPT (The Radio Vibe):
       - ENERGY: High, Joyful, Excited. Like a morning radio show host.
       - LANGUAGE: Simple, Easy English. No big words.
       - OPENING: Start INSTANTLY with a high-energy hook. NO "Asterisk", "Music", or "Sound of...". Just your voice.
       - CONTENT: Explain the tech simply. Use a fun analogy.
       - TONE: Use words like "Whoa", "Imagine this", "Super cool".
       - ENDING: "Check the link, it's wild!"
       - LENGTH: STRICTLY under 45 seconds spoken.
    
    OUTPUT FORMAT:
    ---TELEGRAM---
    [Your short text]
    ---PODCAST---
    [Your script]
    """
    
    # USE FALLBACK GENERATOR HERE
    text = generate_with_fallback(prompt)
    
    if not text:
        return None
        
    try:
        if "---TELEGRAM---" not in text or "---PODCAST---" not in text:
            logger.warning("⚠️ AI response missing markers")
            return None
            
        parts = text.split("---PODCAST---")
        telegram_txt = parts[0].replace("---TELEGRAM---", "").strip()
        podcast_txt = parts[1].strip()
        
        # FINAL CLEANUP
        podcast_txt = re.sub(r'\*.*?\*', '', podcast_txt)
        podcast_txt = re.sub(r'\[.*?\]', '', podcast_txt)
        podcast_txt = re.sub(r'\(.*?\)', '', podcast_txt)
        
        return {
            "telegram": f"{telegram_txt}\n\n🔗 {video_url}",
            "podcast": podcast_txt
        }
    except Exception as e:
        logger.error(f"Parsing error: {e}")
        return None

async def generate_audio(script):
    """Generate audio with Guy (Radio Host)"""
    try:
        voice = "en-US-GuyNeural" 
        communicate = edge_tts.Communicate(script, voice, rate="+10%", pitch="+0Hz")
        await communicate.save("story.mp3")
        return True
    except Exception as e:
        logger.error(f"❌ TTS failed: {e}")
        return False

async def send_to_telegram(telegram_msg, audio_file):
    """Send to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": telegram_msg,
            "parse_mode": "Markdown", 
            "disable_web_page_preview": False
        }
        
        response = requests.post(url, data=data)
        
        if response.status_code == 400:
            logger.warning("⚠️ Markdown failed. Resending as plain text...")
            data["parse_mode"] = None 
            response = requests.post(url, data=data)
            
        response.raise_for_status()
        
        if audio_file and Path(audio_file).exists():
            with open(audio_file, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVoice",
                    data={"chat_id": TELEGRAM_CHAT_ID},
                    files={"voice": f}
                )
            Path(audio_file).unlink()
        
        return True
    except Exception as e:
        logger.error(f"❌ Telegram error: {e}")
        return False

# --- MAIN LOOP ---
async def main():
    logger.info("🚀 Starting AI News Anchor...")
    
    history = load_history()
    new_videos = []
    
    feed_urls = [f.strip() for f in YOUTUBE_FEEDS if f.strip()]
    
    for feed_url in feed_urls:
        try:
            feed = feedparser.parse(feed_url)
            if not feed.entries:
                continue
                
            # Only check the LATEST video from each channel to save API calls
            entry = feed.entries[0]
            vid_id = entry.yt_videoid
            
            if vid_id not in history:
                new_videos.append({
                    "id": vid_id,
                    "title": entry.title,
                    "url": entry.link,
                    "channel": feed.feed.title
                })
        except Exception as e:
            logger.error(f"❌ Feed error for {feed_url}: {e}")
    
    logger.info(f"📺 Found {len(new_videos)} new videos")
    
    for video in new_videos:
        logger.info(f"\n🎬 Processing: {video['title']}")
        
        transcript = get_transcript(video["id"])
        if not transcript:
            continue
        
        content = analyze_video(transcript, video["channel"], video["title"], video["url"])
        if not content:
            continue
        
        audio_ok = await generate_audio(content["podcast"])
        
        success = await send_to_telegram(content["telegram"], "story.mp3" if audio_ok else None)
        
        if success:
            history.add(video["id"])
            save_history(history)
            logger.info(f"✅ Delivered: {video['title']}")
            time.sleep(2)
    
    logger.info("\n✨ Processing complete!")

if __name__ == "__main__":
    asyncio.run(main())