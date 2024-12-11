from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import random
import response_dict

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
    
    # 最も高い確率の感情を取得
    max_index = np.argmax(prob)
    top_emotion = emotion_names_jp[max_index]
    second_emotion = emotion_names_jp[np.argsort(prob)[-2]]

    #print(f"Top emotion: {top_emotion}")

    # ランダムに応答を選択
    responses = response_dict.get(top_emotion, ["その気持ちをもっと話してみてください。"])
    response = random.choice(responses)

    #複数の感情を含む応答を選択
    if top_emotion == "喜び" and second_emotion == "悲しみ":
        responses = response_dict.get("喜び+悲しみ", ["その気持ちをもっと話してみてください。"])
        response = random.choice(responses)
        print(response)

    # numpy.float32をfloatに変換
    #out_dict = {n: float(p) for n, p in zip(emotion_names_jp, prob)}
    out_dict = {"1位の感情":top_emotion,"2位の感情":second_emotion}

    print(out_dict, response)
    print(prob)
    print(response_dict["喜び"])
    return out_dict, response

analyze_emotion("今日はいい天気ですね！")


#ランダムで慰め以外のお叱りを返す





