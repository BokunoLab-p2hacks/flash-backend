from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import random
import os
import subprocess

# 仮想環境の作成、アクティベート、および依存関係のインストール
def setup_environment():
    venv_path = "./myenv"
    if not os.path.exists(venv_path):
        print("仮想環境を作成中...")
        subprocess.run(["python3.11", "-m", "venv", venv_path], check=True)

    activate_script = os.path.join(venv_path, "Scripts", "activate") if os.name == "nt" else os.path.join(venv_path, "bin", "activate")

    if not os.path.exists(activate_script):
        raise FileNotFoundError("仮想環境が正しく作成されていません。")

    print("仮想環境を有効化中...")
    # 仮想環境の有効化をシミュレートして必要なコマンドを実行
    pip_install_command = f"source {activate_script} && pip install -r requirements.txt"
    subprocess.run(pip_install_command, shell=True, check=True)

setup_environment()

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

    # 応答メッセージの辞書（複数の応答文）
    response_dict = {
        "喜び": [
            "すごいですね！その気持ちをもっと広げていきましょう！",
            "素晴らしいです！ぜひこの瞬間を楽しんでください！",
            "その喜びを大切にしてください！"
        ],
        "悲しみ": [
            "辛かったですね。でも、次に進むことで明るい未来が待っていますよ。",
            "悲しい気持ちもいつか和らぎます。焦らずいきましょう。",
            "今はつらいですが、少しずつ前に進めば大丈夫です。"
        ],
        "驚き": [
            "それはびっくりですね。でも、落ち着いて考えてみると意外と良い面もあるかもしれません。",
            "驚きは成長のチャンスです。次の一手を考えてみましょう。",
            "驚くのは自然なことです。少しずつ状況を整理しましょう。"
        ],
        "怒り": [
            "怒りはエネルギーです。次のステップに活かせるチャンスかもしれません。",
            "怒りをうまく利用して、建設的な行動に繋げましょう。",
            "その怒りの原因を整理することで、新しい道が見えてくるかもしれません。"
        ],
        "恐れ": [
            "恐れる気持ちは大切です。でも一歩進めば、新しい景色が見えるかもしれません。",
            "恐怖を乗り越えることで、大きな成長が待っています。",
            "少しずつで構いません。一歩ずつ進んでいきましょう。"
        ],
        "嫌悪": [
            "嫌な気持ちになりましたね。次は自分を守る方法を考えてみましょう。",
            "その感情を大切にして、適切な距離を保ちましょう。",
            "嫌な思いをした時は、自分の気持ちを優先してください。"
        ],
        "信頼": [
            "その信頼感を大切にして、前進してください！",
            "信頼は大切な絆です。そのまま進みましょう。",
            "信頼を基盤に、新しい挑戦をしてみてください！"
        ]
    }

    # ランダムに応答を選択
    responses = response_dict.get(top_emotion, ["その気持ちをもっと話してみてください。"])
    response = random.choice(responses)

    # numpy.float32をfloatに変換
    #out_dict = {n: float(p) for n, p in zip(emotion_names_jp, prob)}
    out_dict = {"1位の感情":top_emotion,"2位の感情":second_emotion}
    return out_dict, response


# FastAPIアプリケーションの作成
app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて許可するオリジンを指定
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
@app.post("/analyze_emotion/")
async def analyze_emotion_api(request: EmotionRequest):
    result = analyze_emotion(request.text)
    return {"input_text": request.text, "emotion_probabilities": result}
