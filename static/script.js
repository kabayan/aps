import * as pdfjsLib from 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.5.136/pdf.min.mjs';

// PDF.js のワーカーを設定
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.5.136/pdf.worker.min.mjs';

// グローバル変数を定義
let currentPdf = null; // 現在表示中のPDFドキュメント
let currentPage = 1; // 現在表示中のページ番号
let currentFilename = null; // 現在表示中のPDFファイル名
let searchResults = null; // 検索結果を格納する変数

/**
 * デバッグログを出力する関数
 * @param {string} message ログメッセージ
 */
function debugLog(message) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [DEBUG] ${message}`);
}

/**
 * ローディング画面を表示する関数
 */
function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}
function showLoadingWithMsg(msg) {
    document.getElementById('loading-overlay').style.display = 'flex';
    document.getElementById('loading-overlay-msg').textContent = msg;
}

/**
 * ローディング画面を非表示にする関数
 */
function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// PDFアップロードフォームの submit イベントハンドラ
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault(); // フォームのデフォルト動作をキャンセル
    debugLog('Upload form submitted');

    const fileInput = document.getElementById('pdf-file');
    const files = fileInput.files; // 選択されたすべてのファイルを取得

    if (files.length === 0) {
        alert('少なくとも1つのファイルを選択してください。');
        return;
    }

    showLoading(); // ローディング画面を表示

    const successFiles = [];
    const failedFiles = [];

    try {
        for (let i = 0; i < files.length; i++) {
            const formData = new FormData();
            formData.append('file', files[i]);

            // '/upload' エンドポイントに fetch API を使用して POST リクエストを送信
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            // レスポンスステータスコードが成功の場合
            if (response.ok) {
                successFiles.push(files[i].name); // 成功したファイル名を記録
                debugLog(`File ${files[i].name} uploaded successfully`);
            } else {
                failedFiles.push(files[i].name); // 失敗したファイル名を記録
                debugLog(`Upload failed for ${files[i].name}`);
            }
        }

        await loadExistingPdfList(); // アップロードされたファイルを含むように既存のPDFリストを更新
        fileInput.value = ''; // ファイル入力フィールドをクリア
    } finally {
        hideLoading(); // ローディング画面を非表示

        // アップロード結果をまとめて表示
        let message = 'アップロード結果:\n\n';

        if (successFiles.length > 0) {
            message += `成功: ${successFiles.join(', ')}\n`;
        }

        if (failedFiles.length > 0) {
            message += `失敗: ${failedFiles.join(', ')}\n`;
        }

        alert(message); // まとめてメッセージを表示
    }
});

/**
 * サーバーから既存のPDFファイルリストを取得し、表示を更新する関数
 */
async function loadExistingPdfList() {
    debugLog('Loading existing PDF list');

    // '/files' エンドポイントに fetch API を使用して GET リクエストを送信
    const response = await fetch('/files');
    const data = await response.json(); // レスポンスをJSON形式でパース

    // 既存のPDFリストを表示する要素を取得
    const pdfList = document.getElementById('existing-pdf-list');
    pdfList.innerHTML = ''; // リストをクリア

    // ファイルが存在する場合
    if (data.files.length > 0) {
        debugLog(`Loaded ${data.files.length} files`);
        // 取得したファイルリストをループ処理
        data.files.forEach(file => {
            // 各ファイルに対してリストアイテムを作成し、リストに追加
            const item = createPdfListItem(file, true);
            pdfList.appendChild(item);
        });
    } else {
        // ファイルが存在しない場合
        debugLog('No files uploaded yet');
        pdfList.innerHTML = '<p>No files uploaded yet.</p>';
    }

    // 削除ボタンの状態を更新
    updateDeleteButtonState();
}

/**
 * PDFファイルリストのアイテムを作成する関数
 * @param {string} file ファイル名
 * @param {boolean} isExistingList 既存のPDFリストに追加するかどうか
 * @returns {HTMLDivElement} 作成されたリストアイテム
 */
function createPdfListItem(file, isExistingList) {
    // リストアイテム要素を作成
    const item = document.createElement('div');
    item.className = 'pdf-item';

    // 既存のPDFリストに追加する場合
    if (isExistingList) {
        // チェックボックスを作成し、リストアイテムに追加
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'pdf-checkbox';
        checkbox.name = 'pdfs';
        checkbox.value = decodeURIComponent(file);
        item.appendChild(checkbox);
    }

    // ファイル名を表示する要素を作成し、リストアイテムに追加
    const span = document.createElement('span');
    span.textContent = decodeURIComponent(file);
    span.className = 'pdf-name';
    item.appendChild(span);

    // 作成したリストアイテムを返す
    return item;
}

/**
 * 削除ボタンの状態を更新する関数
 */
function updateDeleteButtonState() {
    debugLog('Updating delete button state');

    // 削除ボタンとチェックされたファイルを取得
    const deleteButton = document.getElementById('delete-button');
    const checkedFiles = document.querySelectorAll('#existing-pdf-list input:checked');

    // チェックされたファイルがない場合は削除ボタンを無効化、そうでない場合は有効化
    deleteButton.disabled = checkedFiles.length === 0;
    debugLog(`Delete button ${deleteButton.disabled ? 'disabled' : 'enabled'}`);
}

// 削除ボタンのクリックイベントハンドラ
document.getElementById('delete-button').addEventListener('click', async () => {
    debugLog('Delete button clicked');

    // チェックされたファイル名を取得
    const selectedFiles = Array.from(document.querySelectorAll('#existing-pdf-list input:checked')).map(input => input.value);

    // チェックされたファイルがない場合
    if (selectedFiles.length === 0) {
        debugLog('No files selected for deletion');
        alert('No files selected');
        return;
    }

    debugLog(`Deleting files: ${selectedFiles.join(', ')}`);

    // FormData オブジェクトを作成し、削除するファイル名を追加
    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));

    showLoading(); // ローディング画面を表示

    try {
        // '/delete' エンドポイントに fetch API を使用して POST リクエストを送信
        const response = await fetch('/delete', {
            method: 'POST',
            body: formData
        });

        // レスポンスステータスコードが成功の場合
        if (response.ok) {
            const result = await response.json();
            debugLog('Files deleted successfully');
            alert(result.message); // 削除成功メッセージを表示
            await loadExistingPdfList(); // 削除されたファイルを除外するように既存のPDFリストを更新
            clearPdfViewer(); // PDFビューアをクリア
        } else {
            // レスポンスステータスコードがエラーの場合
            const errorData = await response.json();
            debugLog('Deletion failed');
            alert(`Deletion failed: ${errorData.detail}`); // 削除失敗メッセージを表示
        }
    } catch (error) {
        console.error('Error deleting files:', error);
        alert('An error occurred while deleting files'); // エラーメッセージを表示
    } finally {
        hideLoading(); // ローディング画面を非表示
    }
});

/**
 * PDFビューアをクリアする関数
 */
function clearPdfViewer() {
    debugLog('Clearing PDF viewer');

    // キャンバスをクリア
    const canvas = document.getElementById('pdf-viewer');
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);

    // ページ番号をリセット
    document.getElementById('page-number').value = 1;

    // グローバル変数をリセット
    currentPdf = null;
    currentPage = 1;
    currentFilename = null;

    // ダウンロードボタンを無効化
    document.getElementById('download-button').disabled = true;

    // 総ページ数をクリア
    updateTotalPages(0);
}

// 既存のPDFリストのクリックイベントハンドラ
document.getElementById('existing-pdf-list').addEventListener('click', (e) => {
    // クリックされた要素がチェックボックスの場合
    if (e.target.className === 'pdf-checkbox') {
        debugLog(`Checkbox clicked. Checked: ${e.target.checked}`);
        updateDeleteButtonState(); // 削除ボタンの状態を更新
    }
});

// 検索結果のPDFリストのクリックイベントハンドラ
async function handlePdfSelection(e) {
    e.stopPropagation(); // イベントの伝播を停止

    if (e.target.className === 'pdf-name' || (e.target.className === 'pdf-item' && e.target !== e.currentTarget)) {
        const filename = e.target.className === 'pdf-name' ? e.target.textContent : e.target.querySelector('.pdf-name').textContent;
        debugLog(`Selecting PDF from search results: ${filename}`);

        // if (currentFilename === filename) {
        //     debugLog('PDF already selected, skipping load');
        //     return; // 既に選択されているPDFの場合、処理をスキップ
        // }

        // showLoading();
        showLoadingWithMsg("Prepare PDF...")

        try {
            await loadPdf(filename);
            updateUIAfterPdfLoad(e.target.closest('.pdf-item'), filename);
        } catch (error) {
            console.error('Error handling PDF selection:', error);
            debugLog(`Error handling PDF selection: ${error.message}`);
            alert(`Error handling PDF selection: ${error.message}`);
        } finally {
            hideLoading();
        }
    }
}

function updateUIAfterPdfLoad(selectedItem, filename) {
    document.querySelectorAll('.pdf-item').forEach(item => item.classList.remove('selected'));
    selectedItem.classList.add('selected');
    currentFilename = filename;
    document.getElementById('download-button').disabled = false;
}

// イベントリスナーの設定
document.getElementById('search-pdf-list').addEventListener('click', handlePdfSelection);

// リストのフラット化
function flattenAndDeduplicate(data) {
    const flatten = (item) => {
        if (Array.isArray(item)) {
            return item.flatMap(flatten);
        } else if (item instanceof Set) {
            return Array.from(item).flatMap(flatten);
        } else if (typeof item === 'object' && item !== null) {
            return Object.values(item).flatMap(flatten);
        } else {
            return [item];
        }
    };

    const flattened = flatten(data);
    const deduplicatedSet = new Set(flattened);
    return Array.from(deduplicatedSet)
        .filter(item => item !== undefined && item !== null)
        .map(item => {
            if (typeof item === 'string' && item.endsWith('。')) {
                return item.slice(0, -1);
            }
            return item;
        });
}

// グローバル変数
let pdfLoadPromise = null;

/**
 * 指定されたPDFファイルを読み込み、ビューアに表示する関数
 * @param {string} filename PDFファイル名
 */

async function loadPdf(filename) {
    filename = encodeURIComponent(filename)
    debugLog(`Entering loadPdf: ${filename}`);

    if (pdfLoadPromise) {
        debugLog(`PDF already loading, returning existing promise`);
        return pdfLoadPromise;
    }

    pdfLoadPromise = (async () => {
        try {
            // FormData オブジェクトを作成
            const formData = new FormData();
            const contents = new Set();

            // 要約用類似文
            let summary_bases = ""

            // 検索結果から該当するファイルのページ情報を取得
            let result_index = -1
            if (searchResults) {
                const decodedFilename = decodeURIComponent(filename);
                result_index = searchResults.findIndex(result => decodeURIComponent(result[0]) === decodedFilename);
                const currentFileResult = searchResults[result_index];

                if (currentFileResult && currentFileResult[1].length > 0) {
                    debugLog(`currentFileResult found for ${filename}`);

                    // 各ページ番号をフォームデータに追加
                    currentFileResult[1].forEach(([page, content]) => {
                        formData.append('pages', page);
                        contents.add(content)
                        summary_bases = summary_bases + "Page:" + page + "\n" + content + "\n"
                        debugLog(`Added page ${page} to formData`);
                    });
                } else {
                    throw new Error('No pages found for the specified PDF in search results');
                }
            } else {
                throw new Error('No search results available');
            }

            // 要約生成
            let summaryResponse = ""
            if (searchResults[result_index][2] === undefined) {
                debugLog('Summary Bases ${summary_bases}');
                debugLog(`Summarizing PDF: ${filename}`);
                showLoadingWithMsg("要約生成中…しばらくお待ちください…")
                summaryResponse = await generateSummary(summary_bases);
                debugLog('Summary  ${summaryResponse}');
                searchResults[result_index][2] = summaryResponse;
            } else {
                summaryResponse = searchResults[result_index][2];
            }

            // displaySummary(summaryResponse);

            showLoadingWithMsg("ハイライト中...")
            debugLog(`Fetch completed for: ${filename}`);

            // 検索キーワードを取得し、フォームデータに追加
            const searchKeywords = getHighlightKeywordForCurrentPage();
            searchKeywords.forEach(keyword => formData.append('search_strings', keyword));
            const contents_list = flattenAndDeduplicate(contents)
            contents_list.forEach(keyword => formData.append('content_strings', keyword));

            debugLog(`Fetching PDF: ${filename}`);
            // const decodedFilename = decodeURIComponent(filename)
            // const response = await fetch(`/pdf/${encodeURIComponent(decodedFilename)}`, {
            const response = await fetch(`/pdf/${filename}`, {
                    method: 'POST',
                body: formData
            });
            debugLog(`Fetch completed for: ${filename}`);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch PDF');
            }

            debugLog(`Getting Blob: ${filename}`);
            const pdfBlob = await response.blob();
            debugLog(`Blob retrieved: ${filename}`);

            const pdfUrl = URL.createObjectURL(pdfBlob);
            const loadingTask = pdfjsLib.getDocument(pdfUrl);
            currentPdf = await loadingTask.promise;
            debugLog(`PDF loaded: ${filename}`);

            currentPage = 1;
            updateTotalPages(currentPdf.numPages);
            await renderPage();

            debugLog(`PDF rendering completed: ${filename}`);

            displaySummary(summaryResponse);

            return currentPdf;
        } catch (error) {
            debugLog(`Error: ${error.message}`);
            throw error;
        } finally {
            debugLog(`Leaving loadPdf: ${filename}`);
            pdfLoadPromise = null;
        }
    })();

    return pdfLoadPromise;
}

async function generateSummary(text) {
    debugLog('Generating summary');
    const response = await fetch('/summary', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            text: text,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            system_prompt: "あなたは優秀な文書要約システムです。与えられたテキストを日本語で400文字以内で要約してください。"
        })
    });

    if (!response.ok) {
        throw new Error('Failed to generate summary');
    }

    const data = await response.json();
    return data.summary;
}

function displaySummary(summary) {
    let summaryTextarea = document.getElementById('summary');
    summaryTextarea.classList.add("active");
    summaryTextarea.value = summary;
}

/**
 * 総ページ数を更新する関数
 * @param {number} totalPages 総ページ数
 */
function updateTotalPages(totalPages) {
    const totalPagesElement = document.getElementById('total-pages');
    totalPagesElement.textContent = `${totalPages}`;
    document.getElementById('page-number').max = totalPages;
}

/**
 * 指定されたページをレンダリングする関数
 */
async function renderPage() {
    debugLog(`Rendering page ${currentPage}`);

    // currentPdf が null の場合は処理を終了
    if (!currentPdf) return;

    // 指定されたページを取得
    const page = await currentPdf.getPage(currentPage);

    // スケールを設定
    const scale = 1.5;

    // ビューポートを取得
    const viewport = page.getViewport({ scale });

    // キャンバスとコンテキストを取得
    const canvas = document.getElementById('pdf-viewer');
    const context = canvas.getContext('2d');

    // コンテナの幅を取得
    const container = document.getElementById('pdf-viewer-container');
    const containerWidth = container.clientWidth;

    // スケールファクターを計算
    const scaleFactor = containerWidth / viewport.width;

    // キャンバスの高さと幅を設定
    canvas.height = viewport.height * scaleFactor;
    canvas.width = containerWidth;

    // レンダリングコンテキストを作成
    const renderContext = {
        canvasContext: context,
        viewport: viewport,
        transform: [scaleFactor, 0, 0, scaleFactor, 0, 0]
    };

    // ページをレンダリング
    await page.render(renderContext);

    // ページ番号入力の値を現在ページに設定
    document.getElementById('page-number').value = currentPage;
    debugLog(`Page ${currentPage} rendered with highlights`);
}

/**
 * 現在のページのハイライトキーワードを取得する関数
 * @returns {string[]} ハイライトキーワードの配列
 */
function getHighlightKeywordForCurrentPage() {
    // 検索入力欄の値を取得し、空白で分割して配列として返す
    const searchInput = document.getElementById('search-input');
    return searchInput.value.trim().split(/\s+/);
}

// 前のページボタンのクリックイベントハンドラ
document.getElementById('prev-page').addEventListener('click', () => {
    debugLog('Previous page button clicked');

    // 現在ページが1ページ目より大きい場合
    if (currentPage > 1) {
        // 現在ページを減らし、前のページをレンダリング
        currentPage--;
        renderPage();
    }
});

// 次のページボタンのクリックイベントハンドラ
document.getElementById('next-page').addEventListener('click', () => {
    debugLog('Next page button clicked');

    // 現在ページが総ページ数より小さい場合
    if (currentPdf && currentPage < currentPdf.numPages) {
        // 現在ページを増やし、次のページをレンダリング
        currentPage++;
        renderPage();
    }
});

// ページ番号入力の変更イベントハンドラ
document.getElementById('page-number').addEventListener('change', (e) => {
    debugLog(`Page number changed to ${e.target.value}`);

    // 入力されたページ番号を取得
    const pageNum = parseInt(e.target.value);

    // 入力されたページ番号が有効な範囲内である場合
    if (currentPdf && pageNum >= 1 && pageNum <= currentPdf.numPages) {
        // 現在ページを更新し、指定されたページをレンダリング
        currentPage = pageNum;
        renderPage();
    }
});

// ダウンロードボタンのクリックイベントハンドラ
document.getElementById('download-button').addEventListener('click', async () => {
    // 現在表示中のPDFファイル名と検索結果が存在する場合
    if (currentFilename && searchResults) {
        const filename = encodeURIComponent(currentFilename)

        debugLog(`Downloading PDF pages from search results: ${filename}`);

        // showLoading(); // ローディング画面を表示
        showLoadingWithMsg("Prepare PDF..."); // ローディング画面を表示

        try {
            // 検索結果から現在表示中のPDFファイルの情報を取得
            const currentFileResult = searchResults.find(result => result[0] === filename);

            // 検索結果が存在しない場合
            if (!currentFileResult || currentFileResult[1].length === 0) {
                throw new Error('No search results found for the current file');
            }

            // FormData オブジェクトを作成
            const formData = new FormData();

            // 重複ページを避けるために Set を使用
            const pages = new Set();
            const contents = new Set();
            let summary = "";

            // 検索結果からすべてのページを追加
            currentFileResult[1].forEach(([page, content]) => {
                pages.add(page.toString());
                contents.add(content)
                summary = summary + "Page:" + page + "\n" + content + "\n"
            });

            // 要約
            formData.append('summary', "要約：\n" + currentFileResult[2] + "\n" + summary)

            // formData にページを追加
            pages.forEach(page => formData.append('pages', page));

            // 検索キーワードを取得し、フォームデータに追加
            const searchKeywords = getHighlightKeywordForCurrentPage();
            searchKeywords.forEach(keyword => formData.append('search_strings', keyword));
            const contents_list = flattenAndDeduplicate(contents)
            contents_list.forEach(keyword => formData.append('content_strings', keyword));

            // '/pdf/{filename}' エンドポイントに fetch API を使用して POST リクエストを送信
            // const response = await fetch(`/pdf/${currentFilename}`, {
            const response = await fetch(`/pdf/${encodeURIComponent(filename)}`, {
                method: 'POST',
                body: formData
            });

            // レスポンスステータスコードがエラーの場合
            if (!response.ok) {
                throw new Error('Failed to download PDF');
            }

            // レスポンスからPDFのBlobを取得
            const blob = await response.blob();

            // BlobのURLを作成
            const url = window.URL.createObjectURL(blob);

            // ダウンロードリンクを作成
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;

            // 日時を yyyymmddhhmmss 形式で取得
            const date = new Date();
            const yyyymmddhhmmss = date.toISOString().slice(0, 19).replace(/[-T:]/g, '');

            // 検索キーワードを取得し、アンダースコアで結合、ファイル名に含められない文字を#に置き換え
            const keywords = searchKeywords.join('_').replace(/[<>:"\/\\|?*\s]/g, '#');

            // ファイル名に日時とキーワードを追加
            const downloadFilename = `${currentFilename.split('.')[0]}_${yyyymmddhhmmss}_${keywords}.pdf`;
            a.download = downloadFilename;

            // ダウンロードリンクをクリック
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);

            debugLog(`PDF pages from search results downloaded successfully for ${currentFilename}`);
        } catch (error) {
            console.error('Error downloading PDF:', error);
            debugLog(`Error downloading PDF: ${error.message}`);
            alert(`Error downloading PDF: ${error.message}`);
        } finally {
            hideLoading(); // ローディング画面を非表示
        }
    } else {
        alert('No file selected or no search results available');
    }
});


/**
 * 指定されたクエリでPDFを検索する関数
 * @param {string} query 検索クエリ
 */
async function searchPdf(query) {
    debugLog(`Searching for: ${query}`);

    // showLoading(); // ローディング画面を表示
    showLoadingWithMsg("Searching...")

    try {
        // '/search?query={query}' エンドポイントに fetch API を使用して GET リクエストを送信
        const response = await fetch(`/search?query=${encodeURIComponent(query)}`);

        // レスポンスステータスコードがエラーの場合
        if (!response.ok) {
            throw new Error('Search failed');
        }

        // レスポンスをJSON形式でパース
        const data = await response.json();
        debugLog(`Search results: ${JSON.stringify(data)}`);

        // 検索結果を変数に格納
        searchResults = data.results;

        // 検索結果を表示するリストを取得
        const searchPdfList = document.getElementById('search-pdf-list');
        searchPdfList.innerHTML = ''; // リストをクリア

        // 検索結果が存在する場合
        if (searchResults && searchResults.length > 0) {
            // 検索結果をループ処理
            for (const [filename, highlights] of searchResults) {
                // 各ファイルに対してリストアイテムを作成し、リストに追加
                const item = createPdfListItem(decodeURIComponent(filename), false);
                searchPdfList.appendChild(item);

                // 各ページの内容をログ出力
                highlights.forEach(([page, contents]) => {
                    contents.forEach(content => {
                        debugLog(`File: ${filename}, Page: ${page}, Content: ${content}`);
                    });
                });

                debugLog(`Added ${filename} to search PDF list`);
            }

            alert(`Found ${searchResults.length} PDF(s)`); // 検索結果の件数を表示
        } else {
            // 検索結果が存在しない場合
            searchPdfList.innerHTML = '<p>No results found.</p>';
            alert('No results found');
        }
    } catch (error) {
        console.error('Error searching PDF:', error);
        debugLog(`Error searching PDF: ${error.message}`);
        alert(`Error searching PDF: ${error.message}`);
    } finally {
        hideLoading(); // ローディング画面を非表示
    }
}

// 検索ボタンのクリックイベントハンドラ
document.getElementById('search-button').addEventListener('click', () => {
    // 検索入力欄の値を取得
    const query = document.getElementById('search-input').value.trim();

    // 検索クエリが入力されている場合
    if (query) {
        searchPdf(query); // PDFを検索
    } else {
        alert('Please enter a search query');
    }
});

// タブボタンのクリックイベントハンドラ
document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', () => {
        // クリックされたタブのIDを取得
        const tab = button.getAttribute('data-tab');

        // すべてのタブボタンとタブコンテンツからアクティブクラスを削除
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        // クリックされたタブボタンと対応するタブコンテンツにアクティブクラスを追加
        button.classList.add('active');
        document.getElementById(tab).classList.add('active');

        if (tab === 'search-tab') {
            document.getElementById('navigation').classList.add('active');

        } else {
            document.getElementById('navigation').classList.remove('active');
        }

    });
});

// ウィンドウのリサイズイベントハンドラ
window.addEventListener('resize', () => {
    // PDFが読み込まれている場合
    if (currentPdf) {
        // ハイライトレイヤーを取得し、ページを再レンダリング
        const highlightLayer = document.querySelector('.highlight-layer');
        renderPage();
    }
});

// ページ読み込み時に既存のPDFリストを読み込み
loadExistingPdfList();
