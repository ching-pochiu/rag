import os
import re
import io
import hashlib
import pickle
import streamlit as st
import pandas as pd
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

st.set_page_config(
    page_title="土木規範 AI 智慧檢索系統 (Google Gemini 版)",
    page_icon="🏗️",
    layout="wide",
)
st.title("🏗️ 土木工程規範問答系統 (Gemini 雙軌對照版)")

# Google API Key
# 不設任何硬寫死的預設值；金鑰只能來自環境變數，避免程式碼被分享/推上 git 時外洩
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    st.error(
        "未偵測到 GOOGLE_API_KEY 環境變數，請先設定後再啟動：\n"
        "例如在終端機執行 `export GOOGLE_API_KEY=你的金鑰` 後再跑 `streamlit run`，"
        "或在 .streamlit/secrets.toml 中設定並改用 st.secrets['GOOGLE_API_KEY']。"
    )
    st.stop()

st.sidebar.header("📁 規範文件上傳")
uploaded_files = st.sidebar.file_uploader(
    "請上傳土木規範 PDF（可多選）：",
    type=["pdf"],
    accept_multiple_files=True
)

# ──────────────────────────────────────────────────────────────
# 表格清理
# ──────────────────────────────────────────────────────────────

def _looks_like_header_row(row: list) -> bool:
    joined = " ".join(c for c in row if c)
    if re.search(r"(SD|SR)\s*\d", joined):
        return False
    if re.search(r"\d+\s*(以上|以下|～)", joined):
        return False
    return True


def _merge_header_rows(header_rows: list, n_cols: int) -> list:
    filled_rows = []
    for row in header_rows:
        row = list(row) + [None] * (n_cols - len(row))
        last = ""
        filled = []
        for cell in row:
            if cell:
                last = cell.strip()
            filled.append(last)
        filled_rows.append(filled)

    merged = []
    for col_idx in range(n_cols):
        parts = []
        for row in filled_rows:
            val = row[col_idx].replace("\n", "")
            if val and val not in parts:
                parts.append(val)
        merged.append(" ".join(parts) if parts else f"欄{col_idx+1}")
    return merged


def _forward_fill_rows(rows: list, n_cols: int) -> list:
    filled = []
    last_values = [""] * n_cols
    for row in rows:
        row = list(row) + [None] * (n_cols - len(row))
        current = []
        for i, cell in enumerate(row):
            text = str(cell).replace("\n", " ").strip() if cell else ""
            if text:
                last_values[i] = text
            current.append(last_values[i] if not text else text)
        filled.append(current)
    return filled


def _upright_only(page):
    """
    只保留正立（未旋轉）的字元，濾掉旋轉 90 度的浮水印/戳章文字
    （例如「本標準由標準檢驗局授權...下載帳號...下載時間...」）。
    這類浮水印常被 pdfplumber 與正常內文一起抽出、順序錯亂，
    不只污染頁碼偵測，也會污染送進向量庫的條文內容。
    非文字物件（線條、矩形，表格框線偵測要用）維持不濾除。
    """
    return page.filter(
        lambda obj: obj.get("object_type") != "char" or obj.get("upright", True)
    )


_CLAUSE_NO_RE = re.compile(
    r"(?m)^\s*([1-9]\d?(?:\.\d+){1,3})\s+"
    r"(?!(?:小時|分鐘|公分|公尺|公斤|mm|cm|kg|kgf|MPa|N/mm|以上|以下|以內|倍|支|組|次|%|％))"
)
_TABLE_NO_RE = re.compile(r"表\s*(\d+)")


def _extract_clause_no(text: str) -> str:
    """
    抓出這段文字裡第一個看起來像規範節號的字串（例如「17.4」「6.3.1」）。
    CNS 規範慣例是節號自成一行、獨立在段落開頭，用這個位置特徵辨識，
    避免誤抓到內文裡「依 7.1(b)之規定」這種引用語句裡的數字。

    PDF 換行有時會巧合把一個純數值（例如「1.5 小時」的「1.5」、化學成分表
    裡「0.060 以下」的「0.060」）也排到行首，跟真正的節號格式撞在一起。
    這裡用兩個規則降低誤判：
    1. 開頭數字限定 1~99（`[1-9]\\d?`，不能是 0 開頭）——CNS 規範的章號
       就是從 1 開始編，量測數值（尤其化學成分百分比）常以 0 開頭，
       這樣可以擋掉「0.060」這類最常見的誤判來源。
    2. 數字後面緊接著常見單位詞（小時、mm、%等）就不算節號，用來擋掉
       「1.5 小時」這種格式恰好撞上的量測值。
    仍非萬無一失，只是啟發式規則。找不到就傳回空字串，畫面上會退回只顯示頁碼。
    """
    m = _CLAUSE_NO_RE.search(text)
    return m.group(1) if m else ""


def _extract_table_no(caption: str) -> str:
    """從表格標題句（如「表 9 1組竹節鋼筋質量之許可差」）抓出表格編號「9」。"""
    m = _TABLE_NO_RE.search(caption)
    return m.group(1) if m else ""


def _find_table_caption(page_text: str, header: list) -> str:
    """
    在頁面原始文字裡找這張表格的標題句（例如 CNS 規範常見的「表 9 1組竹節
    鋼筋質量之許可差」）。表格清理後的 Markdown 只留下欄位名稱（如「稱號、
    許可差、備考」），原文標題句裡「竹節鋼筋」「質量」這類關鍵字會整句遺失，
    導致使用者用這些字眼提問時，語意/關鍵字檢索反而抓不到乾淨的表格版，
    讓格式更亂的整頁原文搶先命中。

    做法：找「表<數字>」後面、緊接著抵達這張表格第一個欄位標題文字之前的
    那一小段文字，當作標題摘要。找不到就傳回空字串，不影響原本行為。
    """
    first_header = next((h for h in header if h), None)
    if not first_header:
        return ""
    # 表格標題跟第一個欄位標題之間常常隔著換行（pdfplumber 逐行抽字），
    # 先把換行正規化成空白再比對，才不會被換行擋住比對不到。
    flat_text = re.sub(r"\s+", " ", page_text)
    pattern = re.compile(r"表\s*\d+.{0,20}?(?=" + re.escape(first_header) + r")")
    m = pattern.search(flat_text)
    return m.group(0).strip() if m else ""


def _extract_clean_tables(page, display_page, page_text=""):
    tables = page.extract_tables()
    if not tables:
        tables = page.extract_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})

    cleaned = []
    for t_idx, table in enumerate(tables, start=1):
        if not table or len(table) < 2:
            continue

        n_cols = max(len(r) for r in table)

        header_row_count = 0
        for row in table:
            if _looks_like_header_row(row):
                header_row_count += 1
            else:
                break
        header_row_count = max(header_row_count, 1)

        header_rows = table[:header_row_count]
        data_rows = table[header_row_count:]

        header = _merge_header_rows(header_rows, n_cols)
        clean_data_rows = _forward_fill_rows(data_rows, n_cols)
        clean_data_rows = [
            r for i, r in enumerate(clean_data_rows)
            if i == 0 or r != clean_data_rows[i - 1]
        ]

        if not clean_data_rows:
            continue

        cleaned.append({
            "page": display_page,
            "t_idx": t_idx,
            "header": header,
            "caption": _find_table_caption(page_text, header),
            "rows": clean_data_rows,
        })
    return cleaned


# ──────────────────────────────────────────────────────────────
# 頁碼偵測：逐頁偵測 + 區段式內插（取代原本的單一全域 offset）
# ──────────────────────────────────────────────────────────────

_FOOTER_PATTERNS = [
    # — 7 — / 一7一 等格式：部分 PDF 字型的 ToUnicode 對照有異常，
    # 視覺上是一條橫線的頁尾符號，抽出來的實際字元卻是中文數字「一」
    # （U+4E00）而非標準破折號，這裡把它也納入允許的「類破折號」字元集
    re.compile(r"[—–\-─一－]{1,3}\s*([1-9]\d{0,3})\s*[—–\-─一－]{1,3}"),
    re.compile(r"第\s*([1-9]\d{0,3})\s*頁"),                        # 第 7 頁
    re.compile(r"^\s*([1-9]\d{0,3})\s*$", re.MULTILINE),            # 獨立一行的純數字
]


def _detect_footer_page_num(text: str):
    """
    依序嘗試多種格式抓印刷頁碼。前提是傳入的 text 已經濾掉旋轉浮水印
    （見 _upright_only），所以直接在全文找「— N —」這類最具辨識度的
    格式即可，不再需要靠「只看最後 300 字」來閃避浮水印污染
    ——那個做法在浮水印比真正頁尾更晚被抽出時反而會抓錯。
    找到多個符合時取最後一個（最靠近頁面底部）。
    """
    for pat in _FOOTER_PATTERNS:
        matches = pat.findall(text)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                continue
    return None


def build_page_mapping(pdf) -> dict:
    """
    逐頁偵測印刷頁碼，而非只在第一頁算一次 offset。
    找不到頁碼的頁面，用「最近的錨點」局部內插，
    如此附錄重新編號、章節間跳頁等狀況都能各自接上正確的區段，
    不會被前面算出的舊 offset 一路拖著錯下去。
    """
    detected = {}
    for idx, page in enumerate(pdf.pages, start=1):
        text = _upright_only(page).extract_text() or ""
        num = _detect_footer_page_num(text)
        if num is not None:
            detected[idx] = num

    page_mapping = {}
    anchors = sorted(detected.items())

    if not anchors:
        # 完全抓不到任何頁碼格式，才退回用實體頁碼
        for idx in range(1, len(pdf.pages) + 1):
            page_mapping[idx] = f"第 {idx} 頁"
        return page_mapping

    for idx in range(1, len(pdf.pages) + 1):
        if idx in detected:
            page_mapping[idx] = f"第 {detected[idx]} 頁"
            continue

        preceding = [a for a in anchors if a[0] <= idx]
        following = [a for a in anchors if a[0] >= idx]
        anchor_idx, anchor_num = preceding[-1] if preceding else following[0]
        local_offset = anchor_idx - anchor_num
        calc_num = idx - local_offset

        page_mapping[idx] = (
            f"第 {calc_num} 頁" if calc_num > 0 else f"第 {idx} 頁 (封面/前言)"
        )

    return page_mapping


def parse_pdf_to_chunks(pdf_file, doc_name, table_index):
    raw_documents = []

    with pdfplumber.open(pdf_file) as pdf:
        page_mapping = build_page_mapping(pdf)

        for pdf_idx, page in enumerate(pdf.pages, start=1):
            clean_page = _upright_only(page)
            page_text = clean_page.extract_text() or ""
            page_label = page_mapping[pdf_idx]

            for tbl in _extract_clean_tables(clean_page, page_label, page_text):
                header_summary = "、".join(h for h in tbl["header"] if h)
                caption_line = f"{tbl['caption']}\n" if tbl["caption"] else ""
                nl_summary = f"{caption_line}本表格說明「{header_summary}」之規範數據對照。\n"
                md_table = f"\n\n**【{page_label} 表格 {tbl['t_idx']} 規範數據對照表】**\n" + nl_summary
                md_table += "| " + " | ".join(tbl["header"]) + " |\n"
                md_table += "| " + " | ".join(["---"] * len(tbl["header"])) + " |\n"
                for row in tbl["rows"]:
                    md_table += "| " + " | ".join(row) + " |\n"

                table_no = _extract_table_no(tbl["caption"])
                raw_documents.append(Document(
                    page_content=md_table,
                    metadata={"page": page_label, "is_table": True, "table_no": table_no}
                ))

                table_index.append({
                    "doc_name": doc_name,
                    "page": page_label,
                    "header": tbl["header"],
                    "rows": tbl["rows"],
                })

            if page_text:
                formatted_text = "\n".join([line.strip() for line in page_text.split("\n") if line.strip()])
                raw_documents.append(Document(
                    page_content=f"--- [ {page_label} 條文內文 ] ---\n" + formatted_text,
                    metadata={"page": page_label, "is_table": False}
                ))

    return raw_documents


# ──────────────────────────────────────────────────────────────
# 快取：改用檔案內容雜湊當 key，避免 Streamlit 用底線參數
# （不列入快取 hash）造成換檔案或重新上傳後仍拿到舊索引/舊頁碼
# ──────────────────────────────────────────────────────────────

def _hash_files(file_contents) -> str:
    h = hashlib.md5()
    for name, data in file_contents:
        h.update(name.encode("utf-8"))
        h.update(data)
    return h.hexdigest()


@st.cache_resource(show_spinner="正在解析 PDF 並建立向量索引...")
def build_hybrid_retrievers_cached(_file_contents, files_hash: str):
    # _file_contents 底線開頭 → 不參與快取 key 計算，只用來實際執行邏輯
    # files_hash 才是真正決定要不要重新計算的依據
    # persist_dir 加上 v5 版本前綴：已移除行政資訊頁過濾，所有頁面都會入索引，
    # 與 v4（會排除行政頁）的解析結果不同，必須換版本號強制重建，
    # 否則會沿用 v4 舊快取、把行政頁又漏掉。
    persist_dir = os.path.join(".chroma_store", "v5_" + files_hash)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # 正規化成單位向量，讓 cosine 距離/相關性分數的計算有意義、
        # 尺度落在預期範圍內，而不是像原本那樣算出 -2 ~ -5 的異常值
        encode_kwargs={"normalize_embeddings": True},
    )

    all_docs = []
    table_index = []
    parse_errors = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    def _parse_all():
        docs_out = []
        for file_name, file_bytes in _file_contents:
            doc_name = os.path.splitext(file_name)[0]
            try:
                docs = parse_pdf_to_chunks(
                    io.BytesIO(file_bytes), doc_name, table_index
                )
            except Exception as e:
                # 單一檔案解析失敗不應該讓整批索引都掛掉，記錄下來跳過即可
                parse_errors.append(f"{file_name}: {e}")
                continue
            for doc in docs:
                doc.metadata["doc_name"] = doc_name
                if doc.metadata.get("is_table"):
                    docs_out.append(doc)
                else:
                    for chunk in text_splitter.split_documents([doc]):
                        # 幫每個切塊標上節號（例如「17.4」），讓引用來源可以精確到
                        # 條文節次，而不是只能標到頁碼——頁碼在不同版本規範裡可能
                        # 改版跳頁，節號通常比較穩定，對規範查詢來說是更可靠的引用。
                        chunk.metadata["clause_no"] = _extract_clause_no(chunk.page_content)
                        docs_out.append(chunk)
        return docs_out

    # all_docs／table_index 跟向量庫一起存在 persist_dir 底下：Streamlit 進程還在時
    # st.cache_resource 本身就會擋掉重算，這份 pickle 快取是為了「重啟應用程式後、
    # 但磁碟上向量庫已經存在」的情況——沒有它，即使 embedding 不用重算，
    # 每次重啟後第一次查詢還是得重新跑一次完整 PDF 解析（大檔案很花時間）才能
    # 重建 BM25 索引跟規則比對用的 table_index。
    docs_cache_path = os.path.join(persist_dir, "docs_cache.pkl")

    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        # 同一批檔案（依內容雜湊判斷）先前已經 embed 過，直接載入現成的向量庫，
        # 不用重跑一次 embedding
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        if os.path.isfile(docs_cache_path):
            with open(docs_cache_path, "rb") as f:
                all_docs, table_index = pickle.load(f)
        else:
            # 升級前建立的舊快取沒有這份 pickle，退回重新解析一次，
            # 解析完順便補寫，下次重啟就不用再解析了
            all_docs = _parse_all()
            with open(docs_cache_path, "wb") as f:
                pickle.dump((all_docs, table_index), f)
    else:
        all_docs = _parse_all()
        if not all_docs:
            raise ValueError("所有 PDF 都解析失敗，沒有可用內容可建立索引：" + "; ".join(parse_errors))
        vectorstore = Chroma.from_documents(
            all_docs, embeddings, persist_directory=persist_dir,
            # 明確指定 cosine 相似度空間，搭配上面正規化過的 embedding，
            # relevance score 才會落在正常、可用固定門檻過濾的範圍
            collection_metadata={"hnsw:space": "cosine"},
        )
        with open(docs_cache_path, "wb") as f:
            pickle.dump((all_docs, table_index), f)

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    bm25_retriever = BM25Retriever.from_documents(all_docs) if all_docs else None
    if bm25_retriever:
        bm25_retriever.k = RETRIEVAL_CANDIDATE_COUNT

    return vectorstore, vector_retriever, bm25_retriever, table_index, parse_errors


# ──────────────────────────────────────────────────────────────
# 檢索融合：Reciprocal Rank Fusion（取代原本 vector+BM25 單純聯集）
# ──────────────────────────────────────────────────────────────

VECTOR_SCORE_THRESHOLD = 0.2  # 低於此相關性分數的向量結果直接捨棄，不管排到第幾名
# 原本 0.35 是還沒有 reranker 時的保守值，靠這道門檻獨力擋掉不相關結果。
# 現在多了 cross-encoder reranker 做最後把關，門檻可以放寬一點，
# 把「這筆到底相不相關」的判斷更多交給 reranker，避免篩太早、
# 讓語意分數沒那麼高但其實有關的候選，連進 reranker 候選池的機會都沒有。
RETRIEVAL_CANDIDATE_COUNT = 15  # 向量與 BM25 各自先撈的候選數，門檻放寬後同步調寬候選池
RERANK_MODEL = "BAAI/bge-reranker-base"
# 原本用 bge-reranker-v2-m3（5.68 億參數、約 2.2GB）在這台機器（僅 7.8GB 記憶體）
# 上實測會把 python 進程直接記憶體榨乾、被系統強制砍掉，連錯誤訊息都來不及印。
# 改用同系列但小很多的 base 版（2.78 億參數，仍支援中文），
# 在資源有限的機器上才跑得起來，代價是排序精準度會比 v2-m3 略遜一籌。
RRF_CANDIDATE_POOL = 15  # 進 reranker 前的候選池大小，故意留寬一點讓 reranker 有得挑
FINAL_TOP_N = 6  # reranker 排序後，真正送進 LLM context 與畫面顯示的筆數
RERANK_SCORE_THRESHOLD = 0.3  # reranker 分數低於此門檻直接捨棄，不硬湊到 FINAL_TOP_N 筆
# 門檻依實測 10 題結果訂出：真正有用、被答案引用的來源分數多半在 0.36 以上
# （最低的合理引用約 0.36～0.68），而明顯不相關的雜訊分數多落在 0.05～0.23，
# 0.3 大致落在兩者中間，能濾掉雜訊又不會誤殺真正有用但分數沒那麼高的來源。


def _is_structured_query(query: str) -> bool:
    return bool(re.search(
        r"(?:SD|SR)\s*\d{2,4}|\d+(?:\.\d+)?|表格|許可差|規範數據|標準編號",
        query,
        re.IGNORECASE,
    ))


def vector_search_filtered(vectorstore, query: str, k: int = 8):
    """
    用帶分數的相似度查詢取代單純的 retriever.invoke()。
    原本 as_retriever(k=8) 不管分數多低都會回傳滿 8 筆，
    導致「只是矮子裡拔高個」的不相關結果（例如查坍度卻抓到
    抗壓強度標準差表，只因為都出現「標準」兩字）也被排進候選名單。
    這裡直接依相關性分數過濾，分數太低就整筆丟棄。
    """
    try:
        scored = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        # 某些 Chroma 版本/距離度量不支援 relevance score，退回原始排序做法
        return vectorstore.similarity_search(query, k=k)

    if not scored:
        return []

    # 防呆：如果分數尺度異常（例如 embedding 未正規化、Chroma 相似度空間
    # 沒設對，導致分數整批落在 [0,1] 之外），依門檻過濾會把全部結果砍光，
    # 反而讓檢索完全失效。這種情況下退回「不過濾，只依排序取前 k 筆」，
    # 至少維持基本可用，而不是安靜地回傳空結果。
    if max(score for _, score in scored) < 0:
        return [doc for doc, _ in scored]

    return [doc for doc, score in scored if score >= VECTOR_SCORE_THRESHOLD]


def rrf_merge(
    doc_lists: list,
    k: int = 60,
    top_n: int = 6,
    weights: list | None = None,
    max_docs_per_page: int = 2,
    allowed_keys: set | None = None,
):
    """
    對多路檢索結果做 Reciprocal Rank Fusion。
    原本的做法是把 vector 和 BM25 的結果直接聯集、順序完全看誰先被加進去，
    等於沒有真正依相關性排序。RRF 會依每份文件在各路結果中的名次給分
    （名次越前分數越高），兩路都排前面的文件會被推到最前面，
    比任何單一路的原始排序都可靠。

    allowed_keys：向量核可清單（以 page_content 為 key）。有傳且非空時，
    最終只會從這個清單裡挑文件，等於加一道「必須有基本向量相關度」的閘門——
    BM25 仍參與排序，但不能再自己夾帶「向量根本沒撈到」的文件進最終結果
    （例如查詢含通用詞「標準」時，BM25 會把修訂日期頁、發行機關頁等
    純字面命中的雜訊拉進來，這些頁向量分數過低、不在核可清單內，會被擋掉）。
    傳 None 或空集合時不啟用閘門，退回原本行為，避免向量整批被門檻砍光時
    最終結果一片空白。
    """
    if weights is None:
        weights = [1.0] * len(doc_lists)
    if len(weights) != len(doc_lists):
        raise ValueError("weights 數量必須與 doc_lists 相同")

    scores = {}
    doc_map = {}
    for docs, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)

    ranked_keys = sorted(scores.keys(), key=lambda kk: scores[kk], reverse=True)

    # 原本這裡有一條規則：同一頁如果「乾淨表格版」跟「整頁原始文字版」同時是
    # 候選，就直接排除原始文字版，理由是它常跟表格內容重複、格式較亂。
    # 但實測發現這條規則太粗暴：有些頁面同時含兩張表，卻只有一張被 pdfplumber
    # 成功辨識成乾淨表格，另一張的內容就「只存在於原始文字版裡」——這種情況下
    # 整頁排除等於把那張沒被辨識到的表格資料也一起丟掉，答案反而從「查得到」
    # 退步成「查無相關規定」。已經改用替代做法解決根本問題：讓表格版的說明文字
    # 帶回原文標題句（見 `_find_table_caption`），使表格版本來就能靠正常的
    # 相關性排序贏過原始文字版，不需要再靠這條規則強制排除。
    selected = []
    page_counts = {}
    for key in ranked_keys:
        # 向量核可閘門：清單非空時，不在清單內的文件（＝向量沒撈到／分數過低，
        # 純靠 BM25 字面命中進來的）直接跳過，不進最終結果。
        if allowed_keys and key not in allowed_keys:
            continue
        doc = doc_map[key]
        page_key = (doc.metadata.get("doc_name"), doc.metadata.get("page"))
        if page_counts.get(page_key, 0) >= max_docs_per_page:
            continue
        selected.append(doc)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


@st.cache_resource(show_spinner="正在載入 Reranker 模型（首次載入較久，之後會沿用快取）...")
def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def rerank_docs(query: str, docs: list, top_n: int) -> list:
    """
    用 cross-encoder 對 RRF 候選重新排序，取代原本直接拿 RRF 分數當最終依據。
    RRF 只看向量/BM25 各自的排名做加權，並不真的理解語意；cross-encoder
    會把 (問題, 候選內文) 當一對句子一起讀，直接算出相關性分數，
    對法規問答這種需要精準比對條文/數據的場景，排序品質通常比 RRF 更可靠。
    """
    if not docs:
        return []
    model = load_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)
    ranked = sorted(docs, key=lambda d: d.metadata["rerank_score"], reverse=True)
    # 分數低於門檻的直接捨棄，不硬湊到 top_n 筆——candidate pool 不夠相關時，
    # 寧可少顯示幾筆，也不要把明顯不相關的雜訊也列進畫面（實測過，這種雜訊
    # 分數通常在 0.2 以下，混在結果裡會讓使用者誤以為系統把不相關內容也當成依據）。
    filtered = [d for d in ranked if d.metadata["rerank_score"] >= RERANK_SCORE_THRESHOLD]
    if not filtered and ranked:
        # 全部候選都低於門檻時，保留分數最高的 1 筆，不要整個 context 開天窗。
        # 實測過：完全沒有來源時，LLM 會誤以為系統沒給任何參考文本，反問
        # 使用者「請提供參考文本」，而不是照 prompt 指示回答「查無相關規定」；
        # 留 1 筆分數最高（雖然仍偏低）的候選，讓 LLM 至少能判斷「這篇跟問題
        # 有沒有關係」，行為更符合預期，也不影響本來就有合格候選的情況。
        filtered = ranked[:1]
    return filtered[:top_n]


def get_retrieval_debug(vectorstore, bm25_retriever, query: str, k: int = 12, weights: list | None = None):
    """回傳每個候選文件的向量分數、BM25 排名與合併後的 RRF 分數，供 debug 用。

    回傳格式為 list of dict，欄位包含：`doc_name`, `page`, `vector_score`, `vector_rank`,
    `bm25_rank`, `rrf_score`。
    """
    if weights is None:
        weights = [1.0, 1.0]

    # vector 檢索（嘗試拿到 score）
    try:
        vector_scored = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        vector_docs = vectorstore.similarity_search(query, k=k)
        vector_scored = [(doc, None) for doc in vector_docs]

    vector_map = {}
    for rank, (doc, score) in enumerate(vector_scored):
        key = doc.page_content
        vector_map[key] = {"doc": doc, "score": score, "rank": rank}

    # BM25 檢索（無 score，僅有排名）
    bm25_list = bm25_retriever.invoke(query) if bm25_retriever else []
    bm25_map = {}
    for rank, doc in enumerate(bm25_list):
        key = doc.page_content
        bm25_map[key] = {"doc": doc, "rank": rank}

    # 所有候選的 key
    all_keys = list(dict.fromkeys(list(vector_map.keys()) + list(bm25_map.keys())))

    results = []
    for key in all_keys:
        v = vector_map.get(key)
        b = bm25_map.get(key)
        doc = (v or b)["doc"]
        v_score = v["score"] if v else None
        v_rank = v["rank"] if v else None
        b_rank = b["rank"] if b else None

        rrf = 0.0
        if v_rank is not None and weights:
            rrf += weights[0] / (60 + v_rank + 1)
        if b_rank is not None and weights:
            rrf += weights[1] / (60 + b_rank + 1)

        results.append({
            "doc_name": doc.metadata.get("doc_name"),
            "page": doc.metadata.get("page"),
            "vector_score": v_score,
            "vector_rank": v_rank,
            "bm25_rank": b_rank,
            "rrf_score": rrf,
            "snippet": doc.page_content[:200].replace("\n", " "),
        })

    results = sorted(results, key=lambda r: r["rrf_score"], reverse=True)
    return results


# ──────────────────────────────────────────────────────────────
# Groundedness 檢查：答案中出現的數字是否真的存在於檢索到的原文中
# ──────────────────────────────────────────────────────────────

_LIST_MARKER_RE = re.compile(r"(?m)^\s*[\(（]?\d{1,2}[\.\)、）]\s*")


def check_groundedness(response: str, context_str: str) -> list:
    """
    粗略檢查：抓出答案中所有數字，比對是否出現在 context 原文裡。
    直接抓答案裡所有數字字元太寬鬆——LLM 自己排版用的項目編號
    （例如「1. ...」「(2) ...」）也會被當成數字比對，這類格式性數字
    在 context 原文裡當然找不到，會造成大量跟規範內容無關的誤報，
    警告太常出現使用者久了會不理會，防護網形同虛設。
    這裡先把「行首的列表編號」從答案文字裡拿掉，再抓數字比對，
    降低這類格式性數字造成的假警報（該行內文真正的數值仍會照常比對）。
    不是嚴謹的事實查核，但能攔住「檢索沒命中、LLM 卻自己編數字」
    這種對規範查詢來說風險最高的幻覺情況。
    回傳答案中「查無對應原文」的數字列表（可能為空）。
    """
    cleaned_response = _LIST_MARKER_RE.sub("", response)
    answer_nums = set(re.findall(r"\d+(?:\.\d+)?", cleaned_response))
    context_nums = set(re.findall(r"\d+(?:\.\d+)?", context_str))
    return sorted(answer_nums - context_nums, key=lambda x: (len(x), x))


def check_citation_pages(response: str, combined_docs: list) -> list:
    """
    答案裡提到的「第 X 頁」，反查是否真的存在於這次送進 context 的文件中。
    只驗證數字防不了「答案的數值查無依據，但引用頁碼本身是編的」這種情況——
    這裡專門抓「引用了一個根本沒被檢索到的頁碼」。
    回傳答案中查無對應來源的頁碼列表（可能為空）。
    """
    cited_pages = set(re.findall(r"第\s*(\d+)\s*頁", response))
    retrieved_pages = set()
    for doc in combined_docs:
        m = re.search(r"第\s*(\d+)\s*頁", doc.metadata.get("page", ""))
        if m:
            retrieved_pages.add(m.group(1))
    return sorted(cited_pages - retrieved_pages, key=int)


# ──────────────────────────────────────────────────────────────
# 規則比對
# ──────────────────────────────────────────────────────────────

_CODE_PATTERN = re.compile(r"(?:SD|SR)\s*\d{2,4}\s*W?", re.IGNORECASE)


def _normalize_code(text: str) -> str:
    return text.replace(" ", "").upper()


def _find_code_column(header, rows):
    for idx, h in enumerate(header):
        if "符號" in h or "牌號" in h:
            return idx

    best_idx, best_score = None, 0
    for idx in range(len(header)):
        score = sum(
            1 for r in rows
            if idx < len(r) and _CODE_PATTERN.search(r[idx])
        )
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx


def _text_overlap_score(query: str, header: list) -> int:
    q = re.sub(r"\s+", "", query)
    h = "".join(header)
    return sum(1 for i in range(len(q) - 1) if q[i:i + 2] in h)


def lookup_grades(query: str, table_index: list):
    codes = _CODE_PATTERN.findall(query)
    query_codes = []
    for c in codes:
        nc = _normalize_code(c)
        if nc not in query_codes:
            query_codes.append(nc)

    if not query_codes:
        return []

    candidates = {c: [] for c in query_codes}
    for entry in table_index:
        header, rows = entry["header"], entry["rows"]
        code_idx = _find_code_column(header, rows)
        if code_idx is None:
            continue
        for row in rows:
            if code_idx >= len(row):
                continue
            cell_codes = {_normalize_code(tok) for tok in _CODE_PATTERN.findall(row[code_idx])}
            for cell_code in cell_codes:
                if cell_code in candidates:
                    candidates[cell_code].append({
                        "grade": cell_code,
                        "doc_name": entry["doc_name"],
                        "page": entry["page"],
                        "header": header,
                        "row": row,
                    })

    results = []
    for matches in candidates.values():
        if not matches:
            continue
        ranked = sorted(
            matches,
            key=lambda m: (_text_overlap_score(query, m["header"]), len(m["header"])),
            reverse=True,
        )
        # 全部候選都保留，最相關的排第一；若同一牌號在多處出現
        # （可能代表不同文件版本或不同章節），一併列出讓使用者自行判斷
        results.append(ranked)
    return results


# ──────────────────────────────────────────────────────────────
# Streamlit 主流程
# ──────────────────────────────────────────────────────────────

if uploaded_files:
    file_contents = [(f.name, f.read()) for f in uploaded_files]
    files_hash = _hash_files(file_contents)

    # 換一批檔案（依內容雜湊判斷）時，清掉上一次查詢殘留在 session_state 的結果，
    # 避免畫面上還顯示著舊檔案的答案與來源。
    if st.session_state.get("query_files_hash") != files_hash:
        st.session_state.pop("query_result", None)
        st.session_state["query_files_hash"] = files_hash

    try:
        vectorstore, vector_retriever, bm25_retriever, table_index, parse_errors = build_hybrid_retrievers_cached(
            file_contents, files_hash
        )
    except Exception as e:
        st.error(f"建立索引失敗，請確認 PDF 檔案是否正常：{e}")
        st.stop()

    if parse_errors:
        st.sidebar.warning("部分檔案解析失敗，已略過：\n" + "\n".join(parse_errors))
    st.sidebar.success("已完成向量與關鍵字檢索索引建立！")

    user_query = st.text_input("請輸入您要查詢的條文或數據問題：", value="")

    # ── 計算階段：只有按下「送出查詢」才真正跑檢索與呼叫 Gemini ──
    # 結果全部存進 session_state，讓後續互動（例如勾選診斷 checkbox）
    # 重跑整個 script 時，仍能從 session_state 取回結果顯示，
    # 不會因為 st.button 只在按下當下回傳 True 而讓結果整段消失、
    # 也不會每次互動就多打一次 Gemini API。
    if st.button("送出查詢") and user_query:
        with st.spinner("正在透過 Gemini 進行檢索與總結..."):
            doc_vec = vector_search_filtered(
                vectorstore, user_query, k=RETRIEVAL_CANDIDATE_COUNT
            )
            doc_bm25 = bm25_retriever.invoke(user_query) if bm25_retriever else []

            # 結構化查詢更重視精確關鍵字；一般問題維持語意與關鍵字等權。
            retrieval_weights = [1.0, 1.8] if _is_structured_query(user_query) else [1.0, 1.0]

            # 向量核可閘門：以通過向量門檻的 doc_vec 當作「有基本向量相關度」的白名單，
            # 讓最終結果只從中挑選，擋掉純靠 BM25 字面命中（例如通用詞「標準」）
            # 但向量根本沒撈到的雜訊頁。doc_vec 為空時傳 None，閘門自動失效、
            # 退回原本行為，避免最終結果一片空白。
            allowed_keys = {d.page_content for d in doc_vec} if doc_vec else None
            rrf_candidates = rrf_merge(
                [doc_vec, doc_bm25],
                top_n=RRF_CANDIDATE_POOL,
                weights=retrieval_weights,
                max_docs_per_page=2,
                allowed_keys=allowed_keys,
            )
            combined_docs = rerank_docs(user_query, rrf_candidates, top_n=FINAL_TOP_N)

            context_str = "\n\n".join([doc.page_content for doc in combined_docs])

            system_prompt = (
                "你是一位專業且嚴謹的土木規範專家。\n"
                "請根據【參考文本】回答問題。請直接精準列出對應的數值，"
                "並嚴格根據文本中標註的【第 X 頁】與表格編號進行引用說明。\n"
                "若【參考文本】中找不到明確可回答問題的數值或條文，"
                "請直接回答「查無相關規定，請人工確認」，不要自行推測或編造數字。\n\n"
                "【參考文本】：\n{context}\n\n"
                "【問題】：\n{question}"
            )

            prompt = ChatPromptTemplate.from_template(system_prompt)

            response = None
            llm_error = None
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.1-flash-lite",
                    google_api_key=API_KEY,
                    temperature=0
                )
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"context": context_str, "question": user_query})
            except Exception as e:
                llm_error = str(e)

            grade_hits = lookup_grades(user_query, table_index)

        # 一次把這輪查詢的所有結果寫進 session_state，供顯示階段讀取。
        st.session_state["query_result"] = {
            "query": user_query,
            "weights": retrieval_weights,
            "combined_docs": combined_docs,
            "context_str": context_str,
            "response": response,
            "llm_error": llm_error,
            "grade_hits": grade_hits,
        }

    # ── 顯示階段：每次重跑都會執行，只要 session_state 裡有結果就顯示 ──
    # 診斷 checkbox 放在這裡（而非按鈕區塊內），所以勾選它觸發重跑時，
    # 上面的按鈕區塊雖然不執行，這裡仍會照常把既有結果與診斷表畫出來。
    if "query_result" in st.session_state:
        res = st.session_state["query_result"]
        combined_docs = res["combined_docs"]
        context_str = res["context_str"]
        response = res["response"]

        # 檢索診斷表（選擇性）：向量分數、BM25 排名、RRF 合併分數。
        # 用當初實際送出查詢的 query 與 weights 重算（不呼叫 LLM，成本低），
        # 確保診斷內容與這輪答案對得起來。
        if st.sidebar.checkbox("顯示檢索分數與來源詳情"):
            debug_rows = get_retrieval_debug(
                vectorstore, bm25_retriever, res["query"],
                k=RETRIEVAL_CANDIDATE_COUNT, weights=res["weights"],
            )
            if debug_rows:
                df_dbg = pd.DataFrame(debug_rows)
                st.sidebar.markdown("**檢索診斷（前 20）**")
                st.sidebar.table(df_dbg.head(20))

        if res["llm_error"]:
            st.error(f"呼叫 Gemini API 失敗：{res['llm_error']}")

        grade_hits = res["grade_hits"]
        tab_answer, tab_rules, tab_sources = st.tabs(
            ["💡 AI 智慧摘要", "🎯 規則比對", "📊 檢索來源"]
        )

        with tab_answer:
            if response:
                st.markdown(response)

                # Groundedness 檢查：答案中的數字若查無原文依據，提醒人工複核
                ungrounded = check_groundedness(response, context_str)
                if ungrounded:
                    st.warning(
                        "⚠️ 以下數字在檢索到的原文中找不到直接依據，"
                        "可能是 AI 推論或誤植，請人工複核：" + "、".join(ungrounded)
                    )

                # 引用頁碼檢查：答案提到的頁碼若不在這次檢索到的文件中，代表引用可能是編的
                bad_citations = check_citation_pages(response, combined_docs)
                if bad_citations:
                    st.warning(
                        "⚠️ 答案引用了「第 "
                        + "、".join(bad_citations)
                        + " 頁」，但這些頁碼並不在這次實際檢索到的內容中，引用可能有誤，請人工複核。"
                    )
            else:
                st.info("這次查詢沒有產生 AI 回答。")

        with tab_rules:
            if grade_hits:
                for candidates in grade_hits:
                    top = candidates[0]
                    st.markdown(f"**鋼筋牌號：{top['grade']}**（來源：{top['doc_name']} {top['page']}）")
                    df = pd.DataFrame({"欄位項目": top["header"], "規範數據值": top["row"]})
                    st.table(df)
                    if len(candidates) > 1:
                        with st.expander(f"同一牌號在其他 {len(candidates) - 1} 處也有出現，可能為不同版本或章節，點此展開比對"):
                            for other in candidates[1:]:
                                st.markdown(f"來源：{other['doc_name']} {other['page']}")
                                st.table(pd.DataFrame({"欄位項目": other["header"], "規範數據值": other["row"]}))
            else:
                st.info("這次查詢沒有偵測到 SD/SR 鋼筋牌號，無規則比對結果。")

        with tab_sources:
            for i, doc in enumerate(combined_docs, start=1):
                rerank_score = doc.metadata.get("rerank_score")
                score_label = f"｜Rerank 分數：{rerank_score:.3f}" if rerank_score is not None else ""
                # 有節號/表號時優先顯示（比頁碼更精確、更不受改版跳頁影響），
                # 都沒有的頁面（例如純段落條文找不到節號起點）才退回只顯示頁碼。
                table_no = doc.metadata.get("table_no")
                clause_no = doc.metadata.get("clause_no")
                if table_no:
                    locator = f"表 {table_no}｜{doc.metadata.get('page')}"
                elif clause_no:
                    locator = f"第 {clause_no} 節｜{doc.metadata.get('page')}"
                else:
                    locator = doc.metadata.get("page")
                st.success(f"📍 引用來源：{doc.metadata.get('doc_name')} ({locator}){score_label}")
                st.markdown(doc.page_content)
                st.divider()
else:
    st.info("👈 請先於左側邊欄上傳土木規範 PDF 檔案。")