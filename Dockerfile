# NVIDIA CUDA 11.5.2 with Ubuntu 20.04をベースイメージとして使用
FROM nvidia/cuda:11.5.2-devel-ubuntu20.04

# 環境変数を設定してデバッグ出力を抑制
ENV DEBIAN_FRONTEND=noninteractive

# システムの更新と必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    # Meilisearch と PDF 処理に必要なパッケージを追加
    poppler-utils \
    libcurl4-openssl-dev \
    lshw \
    # クリーンアップ
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10のインストールと設定
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && apt-get install -y python3-pip python3.10-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip install --upgrade pip \
    # クリーンアップ
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール (不要なものは削除)
RUN pip uninstall -y torch
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

# Meilisearch と PDF 処理に必要なライブラリを追加
RUN pip install meilisearch meilisearch_python_sdk
RUN pip install PyMuPDF
RUN pip install sentence_transformers
RUN pip install fastapi uvicorn python-multipart ollama
RUN pip install tqdm

# 作業ディレクトリの作成
WORKDIR /root/workspace

# デバッグ用の環境変数を設定
# ENV PYTHONUNBUFFERED=1

# ollama のインストールと設定
# RUN curl -fsSL https://ollama.com/install.sh | sh

# Ollamaサービスの可用性を確認するスクリプトをコピー
COPY wait-for-ollama.sh /usr/local/bin/wait-for-ollama.sh
RUN chmod +x /usr/local/bin/wait-for-ollama.sh

# コンテナ起動時に実行されるコマンド
# CMD ["sh", "-c", "python3 aps.py&"]