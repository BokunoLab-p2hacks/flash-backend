# ベースイメージとして Python 3.11 を使用
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# アプリケーションコードとモデルファイルをコピー
COPY ./app /app/app
COPY ./model_checkpoint /app/model_checkpoint
COPY requirements.txt /app

# 仮想環境を作成
RUN python3.11 -m venv /app/myenv

# 仮想環境を有効化して依存関係をインストール
RUN /app/myenv/bin/pip install --no-cache-dir -r requirements.txt

# ポート番号を指定（Cloud Run のデフォルトポートは 8080）
EXPOSE 10000

# コンテナ起動時に実行するコマンドを指定
CMD ["/app/myenv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
