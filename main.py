import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import google.generativeai as genai
from groq import Groq

# --- CẤU HÌNH API (Lấy từ Render) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Cảnh báo nếu thiếu key (để debug)
if not GROQ_API_KEY: print("⚠️ Chưa có GROQ_API_KEY")
if not GOOGLE_API_KEY: print("⚠️ Chưa có GOOGLE_API_KEY")

try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
    
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        # 🔥 OK! GIỮ NGUYÊN 2.5 THEO Ý BẠN 🔥
        model = genai.GenerativeModel('gemini-2.5-flash') 
        
except Exception as e:
    print(f"❌ Lỗi cấu hình API: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chuyển hướng trang chủ vào App
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# Mount thư mục static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# --- LOGIC ---
def whisper_stt(audio_bytes):
    try:
        return groq_client.audio.transcriptions.create(
            file=("input.webm", audio_bytes), 
            model="whisper-large-v3", 
            response_format="text", 
            language="en")
    except Exception as e:
        print(f"❌ Lỗi Whisper: {e}")
        return None

def repair_transcription(raw_text):
    try:
        prompt = f"Act as a Contextual Corrector. Raw: '{raw_text}'. Fix machine errors silently. Flag pronunciation errors with [PRONUNCIATION ERROR: X->Y]. Output final text only."
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Lỗi Repair: {e}")
        return raw_text

def get_examiner_response(history, user_input):
    system = "You are an IELTS Examiner. Format: **Band: [Score]** 📝 [Feedback] ||| [Next Question]"
    prompt = f"{system}\nHISTORY:\n{history}\nUSER: {user_input}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # In lỗi chi tiết ra log của Render để dễ check
        print(f"❌ LỖI GEMINI: {e}")
        return "Error ||| I cannot connect to the brain right now."

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...), history_context: str = Form("")):
    audio_bytes = await file.read()
    
    raw_text = whisper_stt(audio_bytes)
    if not raw_text: return {"user_text_analyzed": "...", "examiner_question": "I didn't hear anything."}
    
    analyzed_text = repair_transcription(raw_text)
    full_reply = get_examiner_response(history_context, analyzed_text)
    
    feedback = ""
    question = full_reply
    if "|||" in full_reply:
        parts = full_reply.split("|||")
        feedback = parts[0].strip()
        question = parts[1].strip() if len(parts) > 1 else ""

    return {
        "user_text_analyzed": analyzed_text,
        "examiner_feedback": feedback,
        "examiner_question": question
    }