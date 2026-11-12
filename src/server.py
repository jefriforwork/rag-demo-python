import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rag import MODEL_NAME, ask, build_store, load_documents, load_model

ROOT = Path(__file__).resolve().parent.parent
PAGE = (ROOT / "public" / "index.html").read_bytes()
PORT = 3211

print(f"載入模型 {MODEL_NAME} …")
model = load_model()
store = build_store(load_documents(), model)
print(f"手冊已轉成向量，共 {len(store)} 則")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE)

    def do_POST(self):
        if self.path != "/ask":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        question = json.loads(body).get("question") or ""
        result = ask(store, question, model)
        payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(fmt % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"RAG demo http://localhost:{PORT}")
    server.serve_forever()
