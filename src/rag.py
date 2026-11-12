import json
import re
from pathlib import Path

import numpy as np
from text2vec import SentenceModel, semantic_search

ROOT = Path(__file__).resolve().parent.parent
# SentenceModel 來自 text2vec，負責把一句話變成向量。
# 這個模型能把中文和英文放在同一套數字裡比較。
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MIN_SCORE = 0.4


def load_documents():
    file = ROOT / "data" / "handbook.json"
    return json.loads(file.read_text(encoding="utf-8"))


def load_model():
    return SentenceModel(MODEL_NAME, device="cpu")


def build_store(documents, model):
    vectors = model.encode([document["text"] for document in documents])
    store = []
    for document, vector in zip(documents, vectors):
        store.append({**document, "vector": vector})
    return store


def search(store, question, model):
    question_vector = model.encode(question)
    matrix = np.stack([entry["vector"] for entry in store])
    hits = semantic_search(question_vector, matrix, top_k=len(store))[0]
    matches = []
    for hit in hits:
        entry = store[hit["corpus_id"]]
        matches.append({
            "title": entry["title"],
            "text": entry["text"],
            "score": float(hit["score"]),
        })
    return matches


def sentences_of(text):
    parts = re.split(r"[。！？!?]+|(?<=[a-zA-Z])\.\s+", text)
    return [part.strip() for part in parts if part.strip()]


def finish_sentence(sentence):
    if re.search(r"[。！？.!?]$", sentence):
        return sentence
    if re.search(r"[\u4e00-\u9fff]", sentence):
        return sentence + "。"
    return sentence + "."


def winning_match(matches):
    if not matches:
        return None
    best = matches[0]
    second_score = matches[1]["score"] if len(matches) > 1 else 0
    if best["score"] < MIN_SCORE or best["score"] <= second_score:
        return None
    return best


def build_prompt(question, match):
    return "\n".join([
        "你是公司助理。只根據下面這段資料回答。資料沒寫到的，就說手冊裡沒有。",
        "You are a company assistant. Answer only from the passage. If it does not say, say the handbook does not cover it.",
        "",
        "資料 / Passage:",
        match["text"],
        "",
        "問題 / Question:",
        question,
    ])


def generate(question, matches, model):
    best = winning_match(matches)
    if best is None:
        return {
            "answer": "手冊裡沒有對得上的資料。The handbook does not cover this.",
            "prompt": "",
            "usedSentences": [],
        }

    sentences = sentences_of(best["text"])
    question_vector = model.encode(question)
    sentence_vectors = model.encode(sentences)
    hits = semantic_search(question_vector, sentence_vectors, top_k=1)[0]
    hit = hits[0]
    used = sentences[hit["corpus_id"]] if hit["score"] >= MIN_SCORE else best["text"]
    answer = finish_sentence(used)

    return {
        "answer": answer,
        "prompt": build_prompt(question, best),
        "usedSentences": [used],
    }


def ask(store, question, model):
    matches = search(store, question, model)
    return {
        "matches": matches,
        **generate(question, matches, model),
    }
