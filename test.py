"""
from dotenv import load_dotenv
import os
import google.generativeai as genai

# 環境変数の読み込み
load_dotenv()

# API-KEYの設定
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')

input_text = "最近体調が優れなくて、何をするにもエネルギーが出ない…。"
question = f"{input_text}の文章に対して肯定的なアドバイスを返してください。"
res = model.generate_content(question)
print(res.text)

"""

from dotenv import load_dotenv
import os
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import random
from response_dict import response_dict

# 環境変数の読み込み
load_dotenv()

# API-KEYの設定
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')


# モデルとトークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained("model_checkpoint")
model_h = AutoModelForSequenceClassification.from_pretrained("model_checkpoint")

emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']

# Softmax関数
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

# 感情分析関数
def analyze_emotion(text: str):
    model_h.eval()
    tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    preds = model_h(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
    top_emotion = emotion_names_jp[np.argmax(prob)]
    second_emotion = emotion_names_jp[np.argsort(prob)[-2]]
    # 値が0.5以上の感情を取得し、配列に格納（配列の中身は感情：値セット）
    #emotions = []
    #for i in range(len(prob)):
    #    if prob[i] >= 0.4:
    #        emotions.append((emotion_names_jp[i], prob[i]))
    responses = response_dict.get(top_emotion, ["その気持ちをもっと話してみてください。"])
    response = random.choice(responses)
    return response, prob, top_emotion, second_emotion

print(analyze_emotion("今日はいい天気ですね！"))

def responseGemini(text: str):
    question = f"{text}の文章に対して肯定的な慰めやアドバイスを返してください。ただし、文章は短く、簡潔にしてください。"
    res = model.generate_content(question)
    return res.text

print(responseGemini("SNSで見た楽しそうな投稿に、なんだか自分だけ置いてけぼり感。"))

