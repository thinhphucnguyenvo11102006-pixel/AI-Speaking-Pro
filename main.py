import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import google.generativeai as genai
from groq import Groq

# --- 1. CẤU HÌNH API (Thông minh hơn) ---
# Thử load file .env nếu đang chạy trên máy tính (cần cài: pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Đã nạp cấu hình từ file .env (Chế độ Local)")
except:
    print("ℹ️ Đang chạy trên Cloud hoặc không có python-dotenv")

# Lấy Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Khởi tạo biến toàn cục
groq_client = None
model = None

# Kiểm tra Key ngay lập tức
if not GROQ_API_KEY:
    print("❌ LỖI NGHIÊM TRỌNG: Thiếu GROQ_API_KEY! App sẽ không nghe được.")
if not GOOGLE_API_KEY:
    print("❌ LỖI NGHIÊM TRỌNG: Thiếu GOOGLE_API_KEY! App sẽ không trả lời được.")

try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Đã kết nối Groq thành công.")
    
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Giữ nguyên model 2.5 theo ý bạn (nhưng khuyến cáo là nó có thể gây lỗi Error)
        model = genai.GenerativeModel('gemini-2.5-flash') 
        print("✅ Đã cấu hình Gemini thành công.")
        
except Exception as e:
    print(f"❌ Lỗi khởi tạo Client: {e}")

# --- 2. KHỞI TẠO SERVER ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# --- 3. LOGIC XỬ LÝ (Có in log chi tiết) ---

def whisper_stt(audio_bytes):
    # Kiểm tra xem Client có sống không
    if not groq_client:
        print("❌ Lỗi: Groq Client chưa được khởi tạo (Do thiếu Key).")
        return None

    try:
        print(f"🎤 Đang gửi {len(audio_bytes)} bytes lên Groq...", flush=True)
        return groq_client.audio.transcriptions.create(
            file=("input.webm", audio_bytes), 
            model="whisper-large-v3", 
            response_format="text", 
            language="en")
    except Exception as e:
        print(f"❌ Lỗi Whisper (API trả về lỗi): {e}", flush=True)
        return None

def repair_transcription(raw_text):
    if not model: return raw_text
    try:
        prompt = f"Act as a Contextual Corrector. Raw: '{raw_text}'. Fix machine errors silently. Flag pronunciation errors with [PRONUNCIATION ERROR: X->Y]. Output final text only."
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Lỗi Repair: {e}")
        return raw_text

def get_examiner_response(history, user_input):
    if not model: return "Error ||| System API Key missing."
    
    system = "You are an IELTS Examiner. Format: **Band: [Score]** 📝 [Feedback] ||| [Next Question]"
    prompt = f"{system}\nHISTORY:\n{history}\nUSER: {user_input}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ LỖI GEMINI: {e}")
        return f"Error ({str(e)}) ||| I cannot connect to the brain right now."

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...), history_context: str = Form("")):
    # 1. Đọc file
    audio_bytes = await file.read()
    print(f"📩 Server nhận file: {len(audio_bytes)} bytes", flush=True)
    
    # Check file rỗng
    if len(audio_bytes) < 100:
        print("⚠️ File quá nhỏ -> Lỗi Mic phía Client")
        return {"user_text_analyzed": "...", "examiner_question": "Microphone error: File is empty."}

    # 2. Xử lý
    raw_text = whisper_stt(audio_bytes)
    
    if not raw_text: 
        print("⚠️ Whisper trả về None -> Không nghe được gì.")
        return {"user_text_analyzed": "...", "examiner_question": "I didn't hear anything. Please check the Server Logs."}
    
    print(f"🗣️ Nghe được: {raw_text}", flush=True)

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
