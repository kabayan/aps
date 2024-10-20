import os
import asyncio
import numpy as np
from typing import List, Optional
from tqdm import tqdm
from datetime import datetime
import re
import hashlib
import logging
import pymupdf

from sentence_transformers import SentenceTransformer
from collections import OrderedDict

# Meilisearchクライアント
from meilisearch_python_sdk import AsyncClient
from meilisearch_python_sdk.models.settings import (
    Embedders,
    UserProvidedEmbedder,
    Faceting,
)

# ロギングの設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from huggingface_hub import snapshot_download

os.environ["TRANSFORMERS_OFFLINE"] = "1"
model_name = "intfloat/multilingual-e5-large"
local_dir = f"/models/huggingface/{model_name}"
if not os.path.exists(local_dir):
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
model_name = local_dir


class E5:
    """
    SentenceTransformersを用いてテキストの埋め込みベクトルを計算するクラス。
    計算結果をキャッシュして処理の高速化を図る。
    """

    def __init__(self):
        # SentenceTransformerモデルの読み込み
        # self.model = SentenceTransformer(model_name)
        self.model = SentenceTransformer(model_name, device="cpu")

        # 埋め込みベクトルのキャッシュ
        self.embeddings_cache = OrderedDict()
        # キャッシュの最大サイズ
        self.cache_size = 1000

    def compute_hash(self, sentence):
        """
        文章のハッシュ値を計算する。
        キャッシュのキーとして使用。
        """
        return hashlib.md5(sentence.encode()).hexdigest()

    async def embeddings(self, sentences, batch_size=32, is_passage=True):
        """
        文章のリストを受け取り、埋め込みベクトルのリストを返す。
        キャッシュに存在する場合はキャッシュから返し、存在しない場合は計算してキャッシュに追加する。
        """
        # 文章のタイプに応じたprefixを追加
        prefix = "passage: " if is_passage else "query: "
        prefixed_sentences = [prefix + s for s in sentences]

        # 結果を格納するリスト
        result_embeddings = []
        # キャッシュにない文章を格納するリスト
        uncached_sentences = []
        # キャッシュにない文章のインデックスを格納するリスト
        uncached_indices = []

        # 各文章について、キャッシュに存在するか確認
        for i, sentence in enumerate(prefixed_sentences):
            sentence_hash = self.compute_hash(sentence)
            if sentence_hash in self.embeddings_cache:
                # キャッシュに存在する場合は、キャッシュから取得
                result_embeddings.append(self.embeddings_cache[sentence_hash])
                # キャッシュの使用順を更新
                self.embeddings_cache.move_to_end(sentence_hash)
            else:
                # キャッシュに存在しない場合は、計算対象に追加
                uncached_sentences.append(sentence)
                uncached_indices.append(i)

        # キャッシュにない文章が存在する場合
        if uncached_sentences:
            # SentenceTransformerモデルを使って埋め込みベクトルを計算
            batch_embeddings = await asyncio.to_thread(
                self.model.encode,
                sentences=uncached_sentences,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            # 計算結果をキャッシュに追加し、結果リストに挿入
            for i, embedding in zip(uncached_indices, batch_embeddings):
                sentence_hash = self.compute_hash(prefixed_sentences[i])
                self.embeddings_cache[sentence_hash] = embedding
                result_embeddings.insert(i, embedding)

                # キャッシュサイズが上限を超えた場合は、最も古い要素を削除
                if len(self.embeddings_cache) > self.cache_size:
                    self.embeddings_cache.popitem(last=False)

        # 埋め込みベクトルのリストをNumPy配列に変換して返す
        return np.array(result_embeddings)


class MelisearchManager:
    """
    Meilisearchのインデックス作成、ドキュメント追加、削除などを管理するクラス。
    """

    def __init__(self, host, api_key, index_name, index_name_chunk):
        # Meilisearchクライアントの初期化
        self.client = AsyncClient(host, api_key)
        # インデックス名
        self.index_name = index_name
        # チャンク用インデックス名
        self.index_name_chunk = index_name_chunk
        # 埋め込みベクトル計算モデル
        self.e5_model = E5()

    async def log_with_timestamp(self, message, level=logging.INFO):
        """
        タイムスタンプ付きのログメッセージを出力する。
        """
        logging.log(level, message)

    async def extract_metadata_from_pdf(self, pdf_path):
        """
        PDFファイルからメタデータ(タイトル、著者、作成日、ページ数)を抽出する。
        """
        try:
            # PyMuPDFを使用してPDFファイルを開く
            doc = await asyncio.to_thread(pymupdf.open, pdf_path)
            # メタデータを取得
            metadata = doc.metadata
            # タイトル、著者、作成日、ページ数を取得
            title = metadata.get("title", "Unknown Title")
            author = metadata.get("author", "Unknown Author")
            creation_date = metadata.get("creationDate", "Unknown Date")
            total_pages = doc.page_count
            return title, author, creation_date, total_pages
        except Exception as e:
            # エラーが発生した場合はログを出力して例外を再スロー
            await self.log_with_timestamp(
                f"メタデータの抽出中にエラーが発生しました: {str(e)}", logging.ERROR
            )
            raise

    async def extract_text_from_pdf_by_page(self, pdf_path):
        """
        PDFファイルからページごとにテキストを抽出する。
        """
        try:
            # PyMuPDFを使用してPDFファイルを開く
            doc = await asyncio.to_thread(pymupdf.open, pdf_path)
            # ページごとのテキストを格納するリスト
            page_texts = []
            # 各ページについてテキストを抽出
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = await asyncio.to_thread(page.get_text, "text")
                # text = await self.normalize_text(text) # これを入れるとハイライトがうまくいかなくなる
                page_texts.append(text)
            return page_texts
        except Exception as e:
            # エラーが発生した場合はログを出力して例外を再スロー
            await self.log_with_timestamp(
                f"テキスト抽出中にエラーが発生しました: {str(e)}", logging.ERROR
            )
            raise

    async def normalize_text(self, text):
        """
        テキストの正規化を行う。
        """
        text = re.sub(r"\u3000", "", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n\s+", "\n ", text)
        text = re.sub(r"·+", "·", text)
        return text.strip()

    async def _split_text_into_chunks(self, text, chunk_size=200):
        """
        テキストを指定されたチャンクサイズで分割する。
        """
        chunks = []
        sentences = re.split("(?<=。)", text)
        current_chunk = ""

        # 文章ごとにループ処理
        for sentence in sentences:
            # 現在のチャンクと文章を結合した長さがチャンクサイズ以下の場合
            if len(current_chunk) + len(sentence) <= chunk_size:
                # 現在のチャンクに文章を追加
                current_chunk += sentence
            else:
                # 現在のチャンクをリストに追加し、新しいチャンクを作成
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        # 最後のチャンクをリストに追加
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def split_text_into_chunks(self, text, chunk_size=200):
        """
        テキストを指定されたチャンクサイズで分割する。
        文章が chunk_size 以下ならそのまま、超える場合は chunk_size ごとに分割する。
        """
        chunks = []
        sentences = re.split("。", text)

        for sentence in sentences:
            if len(sentence) <= chunk_size:
                # 文章が chunk_size 以下ならそのまま追加
                chunks.append(sentence)
            else:
                # 文章が chunk_size を超える場合、chunk_size ごとに分割
                for i in range(0, len(sentence), chunk_size):
                    chunks.append(sentence[i : i + chunk_size])

        return chunks

    def generate_document_id(self):
        """
        ドキュメントIDを生成する。
        """
        return datetime.now().strftime("%Y%m%d%H%M%S%f")

    async def initialize_index(
        self, index_name, index_field="filename", delete=False, use_vector=False
    ):
        """
        Meilisearchのインデックスを初期化する。
        """
        try:
            # インデックスが存在するか
            index = await self.client.get_index(index_name)
            if delete:
                await index.delete_if_exists()
                await self.log_with_timestamp(
                    f"インデックス '{index_name}' を削除しました。"
                )
                # 新しいインデックスを作成
                await self.client.create_index(index_name)
                await self.log_with_timestamp(
                    f"新しいインデックス '{index_name}' を作成しました。"
                )
            else:
                await self.log_with_timestamp(
                    f"インデックス '{index_name}' は存在します。"
                )
            await self.update_index_settings(index_name, index_field, use_vector)

        except Exception as e:
            # インデックスが存在しない場合はログを出力
            await self.log_with_timestamp(
                f"インデックス '{index_name}' は存在しませんでした: {str(e)}"
            )

    async def update_index_settings(self, index_name, index_field, use_vector=False):
        """
        Meilisearchのインデックス設定を更新する。
        """
        try:
            # インデックス設定を更新
            await self.client.index(index_name).update_faceting(
                faceting=Faceting(max_values_per_facet=100000)
            )
            await self.client.index(index_name).update_filterable_attributes(
                [index_field]
            )
            if use_vector:
                # 埋め込みベクトル計算の設定
                await self.client.index(index_name).update_embedders(
                    Embedders(
                        embedders={"image2text": UserProvidedEmbedder(dimensions=1024)}
                    )
                )
            await self.log_with_timestamp(
                f"インデックス '{index_name}' の設定を更新しました。"
            )
        except Exception as e:
            # エラーが発生した場合はログを出力して例外を再スロー
            await self.log_with_timestamp(
                f"インデックス設定の更新中にエラーが発生しました: {str(e)}",
                logging.ERROR,
            )
            raise

    async def delete_documents_by_field(self, fieldname: str, value: str):
        """
        指定されたフィールドの値を持つドキュメントをMeilisearchから削除する。
        """
        try:
            # 元のドキュメントの削除
            await self.log_with_timestamp(
                f"{fieldname} が '{value}' のドキュメントの削除を開始..."
            )
            task = await self.client.index(self.index_name).delete_documents_by_filter(
                filter=f'{fieldname} = "{value}"'
            )
            task_uid = task.task_uid
            chunk_task = await self.client.index(
                self.index_name_chunk
            ).delete_documents_by_filter(filter=f'{fieldname} = "{value}"')
            task_uid_chunk = chunk_task.task_uid

            await self.wait_for_tasks(task_uid, task_uid_chunk)
            return task_uid, task_uid_chunk

        except Exception as e:
            # エラーが発生した場合はログを出力して例外を再スロー
            await self.log_with_timestamp(
                f"ドキュメントの削除中にエラーが発生しました: {str(e)}", logging.ERROR
            )
            raise

    async def get_unique_filenames(self) -> List[str]:
        """
        Meilisearchに登録されているユニークなファイル名の一覧を取得する。
        """
        try:
            await self.log_with_timestamp("登録されたファイル名の一覧を取得中...")

            # facets 検索を実行
            # 本来はpageingで全件取得すべき※
            search_results = await self.client.index(self.index_name).search(
                "", facets=["filename"]
            )

            # ユニークなファイル名を取得
            unique_filenames = list(
                search_results.facet_distribution["filename"].keys()
            )

            await self.log_with_timestamp(
                f"{len(unique_filenames)}件のユニークなファイル名を取得しました。"
            )
            return unique_filenames
        except Exception as e:
            await self.log_with_timestamp(
                f"ファイル名一覧の取得中にエラーが発生しました: {str(e)}", logging.ERROR
            )
            raise

    async def upload_pdf_to_meilisearch(self, pdf_path):
        """
        PDFファイルをMeilisearchにアップロードし、インデックス化する。
        """
        try:
            # インデックスを初期化
            await self.initialize_index(self.index_name)
            await self.initialize_index(self.index_name_chunk)

            # ファイル名を取得
            filename = os.path.basename(pdf_path)

            # メタデータを抽出
            await self.log_with_timestamp("PDFからメタデータの抽出を開始...")
            title, author, creation_date, total_pages = (
                await self.extract_metadata_from_pdf(pdf_path)
            )
            await self.log_with_timestamp("メタデータの抽出が完了しました。")

            # テキストをページごとに抽出
            page_texts = await self.extract_text_from_pdf_by_page(pdf_path)

            # ドキュメントとチャンクを格納するリスト
            documents = []
            chunk_documents = []

            # テキストを処理し、ドキュメントとチャンクを作成
            await self.log_with_timestamp("テキストの処理とドキュメントの準備を開始...")
            for page_num, text in tqdm(
                enumerate(page_texts),
                total=len(page_texts),
                desc="ページの処理",
                unit="ページ",
            ):
                # ドキュメントを作成
                document = {
                    "uid": self.generate_document_id(),
                    "title": title,
                    "author": author,
                    "creation_date": creation_date,
                    "total_pages": total_pages,
                    "page_number": page_num + 1,
                    "content": text,
                    "filename": filename,
                }
                # ドキュメントをリストに追加
                documents.append(document)

                # テキストをチャンクに分割
                chunks = await self.split_text_into_chunks(text)

                # 各チャンクについて処理
                for i, chunk in enumerate(chunks):
                    # 埋め込みベクトルを計算
                    vector = await self.e5_model.embeddings([chunk])
                    vector = vector[
                        0
                    ].tolist()  # Meilisearchとの互換性のためにリストに変換
                    # チャンクから改行と余分なスペースを削除
                    chunk = chunk.replace("\n", " ").strip()
                    # チャンクドキュメントを作成
                    chunk_document = {
                        "id": self.generate_document_id(),
                        "title": title,
                        "author": author,
                        "creation_date": creation_date,
                        "total_pages": total_pages,
                        "page_number": page_num + 1,
                        "content": chunk,
                        "filename": filename,
                        "chunk_index": i,
                        "_vectors": {"image2text": vector},
                    }
                    # チャンクドキュメントをリストに追加
                    chunk_documents.append(chunk_document)

            await self.log_with_timestamp("ドキュメントの準備が完了しました。")

            # Meilisearchにドキュメントをアップロード
            await self.log_with_timestamp(
                "元のドキュメントを Meilisearch にアップロード中..."
            )
            update = await self.client.index(self.index_name).add_documents(documents)
            await self.log_with_timestamp(
                "元のドキュメントのアップロードが完了しました。"
            )

            # Meilisearchにチャンクドキュメントをアップロード
            await self.log_with_timestamp(
                "チャンク化されたドキュメントを Meilisearch にアップロード中..."
            )
            update_chunk = await self.client.index(self.index_name_chunk).add_documents(
                chunk_documents
            )
            await self.log_with_timestamp(
                "チャンク化されたドキュメントのアップロードが完了しました。"
            )

            # タスクIDを取得
            task_uid = update.task_uid
            task_uid_chunk = update_chunk.task_uid
            await self.log_with_timestamp(
                f"アップロードタスクUID (元のドキュメント): {task_uid}"
            )
            await self.log_with_timestamp(
                f"アップロードタスクUID (チャンク): {task_uid_chunk}"
            )

            # タスクの完了を待つ
            await self.wait_for_tasks(task_uid, task_uid_chunk)

            return task_uid, task_uid_chunk
        except Exception as e:
            # エラーが発生した場合はログを出力して例外を再スロー
            await self.log_with_timestamp(
                f"PDFのアップロード中にエラーが発生しました: {str(e)}", logging.ERROR
            )
            raise

    async def wait_for_tasks(
        self, task_uid, task_uid_chunk, max_retries=60, retry_interval=5
    ):
        """
        Meilisearchのタスクが完了するまで待つ。
        """
        status = "succeeded"
        status_chunk = "succeeded"
        # 最大試行回数までループ
        for _ in range(max_retries):
            # タスクの状態を取得
            if task_uid != -1:
                task = await self.client.get_task(task_uid)
                status = task.status
            if task_uid_chunk != -1:
                task_chunk = await self.client.get_task(task_uid_chunk)
                status_chunk = task_chunk.status

            # 両方のタスクが成功した場合
            if status == "succeeded" and status_chunk == "succeeded":
                await self.log_with_timestamp("タスクは正常に完了しました！")
                return
            # いずれかのタスクが失敗した場合
            elif status == "failed" or status_chunk == "failed":
                await self.log_with_timestamp(
                    f"タスクは失敗しました。エラー: {task.error or task_chunk.error}",
                    logging.ERROR,
                )
                raise Exception("タスクの実行に失敗しました")

            # タスクが完了していない場合は、ログを出力して待機
            await self.log_with_timestamp(
                f"タスクの状態: {status}, チャンクタスクの状態: {status_chunk}"
            )
            await asyncio.sleep(retry_interval)

        # タイムアウトした場合
        await self.log_with_timestamp(
            "タスクの完了を待機中にタイムアウトしました", logging.ERROR
        )
        raise TimeoutError("タスクの完了を待機中にタイムアウトしました")


# 使用例
async def main():
    # MeilisearchManagerのインスタンス化
    manager = MelisearchManager(
        "http://localhost:7700", "masterKey", "pdf_documents", "pdf_documents_chunks"
    )
    # 処理対象のPDFファイルパス
    pdf_path = "example.pdf"
    try:
        # PDFファイルをMeilisearchにアップロード
        await manager.upload_pdf_to_meilisearch(pdf_path)
    except Exception as e:
        # エラーが発生した場合はログを出力
        logging.error(f"メイン処理中にエラーが発生しました: {str(e)}")


# メイン関数を実行
if __name__ == "__main__":
    asyncio.run(main())
