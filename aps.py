import os
import asyncio
import logging
from fastapi import FastAPI, File, Response, UploadFile, Form, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
import shutil

from tqdm import tqdm
import pymupdf
from io import BytesIO
from typing import List, Optional
from urllib.parse import quote, unquote
import re

# Meilisearchの管理クラスをインポート
from meilsearch_manager import MelisearchManager
from meilisearch_python_sdk import AsyncClient

from ollama import AsyncClient as OllamaAsyncClient

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

# FastAPI アプリケーションを作成
app = FastAPI()

# ログレベルを設定
logging.basicConfig(level=logging.WARNING)

# Meilisearch の設定
MEILISEARCH_HOST = "http://meliserver:7700"  # Meilisearch サーバーのホスト名
MEILISEARCH_API_KEY = "aSampleMasterKey"  # Meilisearch サーバーのAPIキー
INDEX_NAME = "pdf_documents"  # Meilisearch のインデックス名
INDEX_NAME_CHUNK = "pdf_documents_chunks"  # Meilisearch のチャンク用インデックス名

# MeilisearchManager のインスタンス化
meilisearch_manager = MelisearchManager(
    MEILISEARCH_HOST, MEILISEARCH_API_KEY, INDEX_NAME, INDEX_NAME_CHUNK
)

# Meilisearch クライアントの初期化
client = AsyncClient(MEILISEARCH_HOST, MEILISEARCH_API_KEY)

# 静的ファイルの提供設定
app.mount("/static", StaticFiles(directory="static"), name="static")

# アップロードディレクトリの設定
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# OllamaModel クラスの定義
class OllamaModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.logger = logging.getLogger(f"{__name__}.OllamaModel")
        self.client = OllamaAsyncClient(host=OLLAMA_HOST)

    async def generate_text(self, messages, temperature, top_p, top_k):
        self.logger.info(
            f"Ollamaモデルでテキスト生成を開始 (temperature={temperature}, top_p={top_p}, top_k={top_k})"
        )
        rst = await self.client.chat(
            model=self.model_name,
            messages=messages,
            options={
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "num_predict": 2048,
                "num_ctx": 2048,
                "repeat_penalty": 1.1,
            },
        )
        self.logger.info("Ollamaモデルでのテキスト生成が完了")
        return rst["message"]["content"]


# TextGenerationApp クラスの定義
class TextGenerationApp:
    def __init__(self):
        # self.ollama_model = OllamaModel("dsasai/llama3-elyza-jp-8b")
        # self.ollama_model = OllamaModel("llama3.2:1b")
        # self.ollama_model = OllamaModel("llama3.2:latest")
        self.ollama_model = OllamaModel("lucas2024/gemma-2-2b-jpn-it:q8_0")
        self.logger = logging.getLogger(f"{__name__}.TextGenerationApp")

    async def process_single_input(
        self,
        line: str,
        temperature,
        top_p,
        top_k,
        system_prompt,
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": line},
        ]

        self.logger.info(f"テンプレート展開結果 {messages}")

        prompt = ""

        output = await self.ollama_model.generate_text(
            messages, temperature, top_p, top_k
        )

        self.logger.info(f"単一入力の処理が完了 : {prompt}")
        # return output.replace("\n", "###")
        return output


# TextGenerationApp のインスタンス化
text_generation_app = TextGenerationApp()


# ルートエンドポイント
@app.get("/")
async def read_root():
    # 静的ファイルindex.htmlを返す
    return FileResponse(os.path.join("static", "index.html"))


# PDFアップロードエンドポイント
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    アップロードされたPDFファイルを保存し、Meilisearchにインデックス化する。
    """

    # ファイル名をURLエンコードする
    encoded_filename = quote(file.filename)
    file_path = os.path.join(UPLOAD_DIR, encoded_filename)

    # アップロードされたファイルを保存
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # MeilisearchにPDFファイルをアップロード
    task_uid, task_uid_chunk = await meilisearch_manager.upload_pdf_to_meilisearch(
        file_path
    )

    # Meilisearchのタスクが完了するまで待つ
    while True:
        task = await meilisearch_manager.client.get_task(task_uid)
        task_chunk = await meilisearch_manager.client.get_task(task_uid_chunk)
        if task.status == "succeeded" and task_chunk.status == "succeeded":
            break
        elif task.status == "failed" or task_chunk.status == "failed":
            raise HTTPException(
                status_code=500, detail="Failed to upload PDF to Meilisearch"
            )
        await asyncio.sleep(1)

    # インデックス化が完了したら、成功メッセージを返す
    return {
        "filename": encoded_filename,
        "message": "PDF uploaded and indexed successfully",
    }


# ファイル一覧取得エンドポイント
@app.get("/files")
async def list_files():
    """
    アップロードされているファイルの一覧を取得する。
    Meilisearchに登録されているファイル名と、アップロードディレクトリ内のファイル名を結合して返す。
    """
    try:
        # Meilisearchから登録されているファイル名の一覧を取得
        meilisearch_files = await meilisearch_manager.get_unique_filenames()

        return {"files": meilisearch_files}
    except Exception as e:
        logging.error(f"ファイル一覧の取得中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail="ファイル一覧の取得に失敗しました")


# ファイル削除エンドポイント
@app.post("/delete")
async def delete_files(files: List[str] = Form(...)):
    """
    指定されたファイル名のファイルを削除する。
    Meilisearchのインデックスと、アップロードディレクトリからファイルを削除する。
    """
    fieldname = "filename"
    deleted_files = []
    for file in files:
        try:
            # Meilisearchからドキュメントを削除
            delete_file = quote(file, safe="")
            await meilisearch_manager.delete_documents_by_field(fieldname, delete_file)
            deleted_files.append(file)
        except Exception as e:
            logging.error(
                f"{fieldname} '{file}' の削除中にエラーが発生しました: {str(e)}"
            )

    if deleted_files:
        return {
            "message": f"以下のファイルが正常に削除されました: {', '.join(deleted_files)}"
        }
    else:
        raise HTTPException(status_code=400, detail="ファイルの削除に失敗しました")


# PDFページ取得エンドポイント
@app.post("/pdf/{filename}")
async def get_pdf_pages(request: Request, filename: str):
    """
    指定されたPDFファイルの、指定されたページを結合し、
    指定されたテキストをハイライトしたPDFファイルを返す。
    """
    logging.warning(f"Received request for file: {filename}")

    # ファイル名をエンコード
    # encoded_output_filename = quote(
    #     decoded_filename.encode("latin-1", errors="ignore").decode("latin-1")
    # )
    encoded_output_filename = quote(filename.encode("utf-8"))

    # リクエストのフォームデータを取得
    form_data = await request.form()
    logging.warning(f"Raw form data: {form_data}")

    # フォームデータからページ番号と検索するテキストを取得
    pages = form_data.getlist("pages")
    search_strings = form_data.getlist("search_strings")
    content_strings = form_data.getlist("content_strings")
    summary = form_data.get("summary")
    logging.warning(f"Pages requested (raw): {pages}")
    logging.warning(f"Search strings: {search_strings}")
    logging.warning(f"Content strings: {content_strings}")

    # ファイルパスを作成
    file_path = os.path.join(UPLOAD_DIR, encoded_output_filename)

    # ファイルが存在しない場合は404エラー
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        logging.warning(f"Pages prepare (raw): {pages}")

        # PDF文書を開く
        doc = pymupdf.open(file_path)

        # 出力用のPDF文書を作成
        output_doc = pymupdf.open()

        # サマリーが指定された場合、新しいページを作成して先頭に追加
        if summary:
            # 新しいページを作成
            page = output_doc.new_page(width=595, height=842)  # A4サイズ (ポイント単位)

            # 日本語フォントを指定（システムにインストールされている日本語フォントを使用）
            font = pymupdf.Font("japan")
            fontsize = 11
            line_height = fontsize * 1.4  # 日本語の場合、行間を少し広めに
            margin = 50

            def draw_wrapped_text(page, text, start_y, output_doc):
                x = margin
                y = start_y
                width = (page.rect.width - 2 * margin) * 0.9
                for line in text.split("\n"):
                    words = list(line)  # 日本語の場合、文字ごとに分割
                    line = ""
                    for char in words:
                        test_line = line + char
                        text_width = font.text_length(test_line, fontsize=fontsize)
                        if text_width <= width:
                            line = test_line
                        else:
                            page.insert_text(
                                (x, y), line, fontsize=fontsize, fontname="japan"
                            )
                            y += line_height
                            line = char
                            if y + line_height > page.rect.height - margin:
                                # 新しいページを作成
                                page = output_doc.new_page(width=595, height=842)
                                y = margin
                    if line:
                        page.insert_text(
                            (x, y), line, fontsize=fontsize, fontname="japan"
                        )
                        y += line_height
                    # 段落の後に少し余白を追加
                    y += line_height * 0.5
                    if y + line_height > page.rect.height - margin:
                        page = output_doc.new_page(width=595, height=842)
                        y = margin
                return page, y

            current_y = margin
            page, current_y = draw_wrapped_text(page, summary, current_y, output_doc)

            logging.warning(f"Summary page(s) added")

        # 指定されたページを追加
        if pages:
            for page_str in pages:
                try:
                    # ページ番号を整数に変換 (0スタート)
                    page_num = int(page_str) - 1

                    # ページ番号が有効な範囲内であればページを追加
                    if 0 <= page_num < len(doc):
                        page = doc.load_page(page_num)

                        # 指定されたテキストを検索してハイライト
                        for search_str in search_strings:
                            text_instances = page.search_for(
                                search_str.replace('"', "")
                            )
                            for inst in text_instances:
                                highlight = page.add_highlight_annot(inst)
                                highlight.update()
                        for search_str in content_strings:
                            text_instances = page.search_for(search_str, flags=0)
                            # logging.warning(f"content_strings: {search_str}")

                            for inst in text_instances:
                                highlight = page.add_underline_annot(inst)
                                highlight.update()

                        # ハイライトされたページを新しいPDFに追加
                        output_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    else:
                        logging.warning(f"Page number out of range: {page_num + 1}")
                except ValueError:
                    logging.warning(f"Invalid page number: {page_str}")
        else:
            # ページ指定がない場合は全ページを追加
            for page in doc:
                # 指定されたテキストを検索してハイライト
                for search_str in search_strings:
                    text_instances = page.search_for(search_str)
                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.update()

                output_doc.insert_pdf(doc)

        # 出力用のPDFデータを取得
        output_pdf = output_doc.tobytes()

        # PDF文書を閉じる
        doc.close()
        output_doc.close()

        logging.warning(f"Pages prepare done (raw): {pages}")

        # PDFファイルをストリーミングで返す
        return Response(
            content=output_pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_output_filename}"
            },
        )
    except Exception as e:
        # エラーが発生した場合はログを出力して500エラー
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the PDF: {str(e)}",
        )


# 検索エンドポイント
@app.get("/search")
async def search(query: str):
    """
    Meilisearchを使ってPDFを検索し、結果を返す。
    """
    # Meilisearchで検索
    search_results = await client.index(INDEX_NAME_CHUNK).search(query)

    # 結果を格納する辞書
    result_dict = {}

    # 結果の最大件数
    result_limit = 100

    # 検索結果の表示
    print(search_results.hits[:result_limit])

    # 検索結果をループ処理
    for result in search_results.hits[:result_limit]:
        filename = result["filename"]
        page_number = result["page_number"]
        # 複数の空白と中黒点を1つにまとめる
        content = re.sub(r"\s+", " ", result["content"])  # 複数の空白を1つに
        content = re.sub(r"·+", "·", content)  # 複数の中黒点を1つに
        content = content.strip()  # 先頭と末尾の空白を削除

        # ファイル名が辞書に存在しない場合は、新しい辞書を作成
        if filename not in result_dict:
            result_dict[filename] = {}

        # ページ番号が辞書に存在しない場合は、新しい集合を作成
        if page_number not in result_dict[filename]:
            result_dict[filename][page_number] = set()

        # コンテンツを集合に追加（重複を自動的に排除）
        result_dict[filename][page_number].add(content)

    # 結果を整形
    final_results = {
        "results": [
            [
                filename,
                sorted(
                    [[page, list(contents)] for page, contents in pages.items()],
                    key=lambda x: x[0],  # ページ番号でソート
                ),
            ]
            for filename, pages in result_dict.items()
        ]
    }

    # 整形した結果を表示
    print(final_results)

    # JSON形式で結果を返す
    return JSONResponse(final_results)


@app.post("/summary")
async def generate_summary(
    text: str = Form(...),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    top_k: int = Form(40),
    system_prompt: str = Form(
        "あなたは優秀な文書要約システムです。与えられたテキストを必ず日本語で200文字程度に要約してください。"
    ),
):
    try:
        result = await text_generation_app.process_single_input(
            text, temperature, top_p, top_k, system_prompt
        )
        return JSONResponse(content={"summary": result})
    except Exception as e:
        logging.error(f"要約生成中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail="要約の生成に失敗しました")


# アプリケーションのエントリーポイント
if __name__ == "__main__":
    # Uvicornを使ってアプリケーションを起動
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
