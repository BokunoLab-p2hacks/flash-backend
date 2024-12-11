# ベースイメージ
FROM python:3.11-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 仮想環境を作成
RUN python3.11 -m venv /app/myenv

# 必要な依存ファイルをコピー
COPY requirements.txt /app/
RUN /app/myenv/bin/pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY app /app/app

# モデルチェックポイントをコピー
COPY model_checkpoint /app/model_checkpoint

# 必要なポートを公開
EXPOSE 8000

# アプリケーションを起動
CMD ["sh", "-c", "/app/myenv/bin/uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
