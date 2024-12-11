# ベースイメージを指定
FROM python:3.11-slim

# 環境変数の設定
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 作業ディレクトリを作成
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

# 仮想環境を有効化して依存関係をインストール
RUN /app/myenv/bin/pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . /app/

# 必要なポートを公開
EXPOSE 8000

# 仮想環境を有効化してFastAPIアプリを実行
CMD ["/app/myenv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
