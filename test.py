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