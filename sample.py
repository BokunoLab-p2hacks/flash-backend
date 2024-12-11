from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import random
import os
from response_dict import response_dict  # 修正箇所

# モデルとトークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained("model_checkpoint")
model = AutoModelForSequenceClassification.from_pretrained("model_checkpoint")

emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']

# Softmax関数
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

# 感情分析関数
def analyze_emotion(text: str):
    model.eval()
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    preds = model(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
    top_emotion = emotion_names_jp[np.argmax(prob)]
    responses = response_dict.get(top_emotion, ["その気持ちをもっと話してみてください。"])
    response = random.choice(responses)
    return top_emotion, response, prob

# テスト実行
if __name__ == "__main__":
    print(analyze_emotion("今日はいい天気ですね！"))
