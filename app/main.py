from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import random
from pydantic import BaseModel

# 環境変数の読み込み
load_dotenv()

# API-KEYの設定
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')

# モデルとトークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained("../model_checkpoint")
model_h = AutoModelForSequenceClassification.from_pretrained("../model_checkpoint")

emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']

# Softmax 関数の実装
def np_softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# 感情分析関数
def analyze_emotion(text: str):
    model_h.eval()
    tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    preds = model_h(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0]) 
    top_emotion = emotion_names_jp[np.argmax(prob)]
    second_emotion = emotion_names_jp[np.argsort(prob)[-2]]
    probs = []
    for i in range(len(prob)):
        probs.append((emotion_names_jp[i], prob[i]))
    return probs, top_emotion, second_emotion

print(analyze_emotion("今日はいい天気ですね！"))

def responseGemini(text: str):
    question = f"{text}の文章に対して肯定的な慰めやアドバイスを返してください。ただし、文章は2文以内でお願いします。"
    res = model.generate_content(question)
    return res.text

#print(responseGemini("最近体調が優れなくて、何をするにもエネルギーが出ない…。"))

def analyze_gemini(text: str):
    # 感情分析
    probs, top_emotion, second_emotion = analyze_emotion(text)
    # Gemini
    question = f"{text}の文章に対して肯定的な慰めやアドバイスを返してください。ただし、文章は2文以内でお願いします。"
    res = model.generate_content(question)
    return probs, top_emotion, second_emotion, res.text

#print(analyze_gemini("最近体調が優れなくて、何をするにもエネルギーが出ない…。"))

# FastAPIアプリケーションの作成
app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストモデル
class EmotionRequest(BaseModel):
    text: str

# エンドポイント: ルート
@app.get("/")
async def root():
    return {"message": "Welcome to the Emotion Analysis API"}

# エンドポイント: 感情分析
@app.post("/analyze_emotion")
async def analyze_emotion_endpoint(request: EmotionRequest):
    text = request.text
    response, top_emotion, second_emotion = analyze_emotion(text)
    return {
        "text": text,
        "response": response,
        "top_emotion": top_emotion,
        "second_emotion": second_emotion
    }

# エンドポイント: Gemini
@app.post("/gemini")
async def responseGemini_endpoint(request: EmotionRequest):
    text = request.text
    response = responseGemini(text)
    return {
        "text": text,
        "response": response
    }

# エンドポイント: 感情分析+Gemini
@app.post("/analyze_gemini")
async def analyze_gemini_endpoint(request: EmotionRequest):
    text = request.text
    probs, top_emotion, second_emotion, response = analyze_gemini(text)
    return {
        "text": text,
        "response": response,
        "probability": probs,
        "top_emotion": top_emotion,
        "second_emotion": second_emotion
    }

# エンドポイント: 感情、投稿の傾向に合わせた応答
@app.post("/moveMatch_response")
async def moveMatch_response():
    return {"message": "投稿頻度、感情の動きに合わせた応答を返します"}
