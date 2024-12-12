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
tokenizer = AutoTokenizer.from_pretrained("/app/model_checkpoint")
model_h = AutoModelForSequenceClassification.from_pretrained("/app/model_checkpoint")
#ローカルパス
#tokenizer = AutoTokenizer.from_pretrained("../model_checkpoint")
#model_h = AutoModelForSequenceClassification.from_pretrained("../model_checkpoint")


emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
emotion_names_en = ['joy', 'sadness', 'anticipation', 'surprise', 'anger', 'fear', 'disgust', 'trust']

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
    return [(emotion, float(prob)) for emotion, prob in zip(emotion_names_en, prob)]

#print(analyze_emotion("親戚のおじさんが亡くなりました。"))

def responseGemini(text: str):
    question = f"{text}の文章に対して肯定的な慰めやアドバイスを返してください。"
    res = model.generate_content(question)
    return res.text

#print(responseGemini("最近体調が優れなくて、何をするにもエネルギーが出ない…。"))

def response_scold(text: str):
    question = f"{text}の文章に対してお叱りをしてください。"
    res = model.generate_content(question)
    return res.text

def response_praise(text: str):
    question = f"{text}の文章に対して褒め言葉をしてください。"
    res = model.generate_content(question)
    return res.text

def filtaring_emotion(text: str):
    question = f"{text}の内容が誹謗中傷や差別的な表現、死、病気、体調などの意味を含む場合は-1、含まない場合は1を返してください。"
    res = model.generate_content(question)
    return res.text

#print(filtaring_emotion("忘れ物して会社に戻ったら、めちゃくちゃ怒られた。自分が情けない。"))

def analyze_gemini(text: str):
    # 感情分析
    probs = analyze_emotion(text)
    # Gemini
    question = f"{text}の文章に対して肯定的な慰めやアドバイスを返してください。"
    res = model.generate_content(question)
    probs = [(emotion, float(prob)) for emotion, prob in probs]
    return probs, res.text

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
    response = analyze_emotion(text)
    response_dict = {emotion: prob for emotion, prob in response}
    return {
        "text": text,
        "response": response,
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
    probs = analyze_gemini(text)
    probs_dict = {emotion: prob for emotion, prob in response}
    response = responseGemini(text)
    return {
        "text": text,
        "response": response,
        "probability": probs,
    }

# エンドポイント: 感情、投稿の傾向に合わせた応答
@app.post("/scold_response")
async def response_scold_endpoint(request: EmotionRequest):
    text = request.text
    response = response_scold(text)
    return {
        "text": text,
        "response": response
    }

@app.post("/praise_response")
async def response_praise_endpoint(request: EmotionRequest):
    text = request.text
    response = response_praise(text)
    return {
        "text": text,
        "response": response
    }