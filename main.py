from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import time
from ingest import ingest
from query import retrieve, rerank, answer

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Mini RAG App</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        textarea { width: 100%; height: 120px; }
        input[type=text] { width: 100%; padding: 8px; }
        button { padding: 10px 20px; margin-top: 10px; }
        .box { border: 1px solid #ccc; padding: 15px; margin-top: 20px; }
        .source { font-size: 14px; color: #555; margin-top: 5px; }
        #upload-status { color: green; margin-top: 10px; }
    </style>
</head>
<body>

<h2>Mini RAG Demo</h2>

<div class="box">
<h3>1. Upload / Paste Text</h3>
<textarea id="upload-text" placeholder="Paste document text here"></textarea>
<br>
<button onclick="uploadText()">Upload</button>
<div id="upload-status"></div>
</div>

<div class="box">
<h3>2. Ask Question</h3>
<form action="/ask" method="post">
    <input type="text" name="question" placeholder="Ask a question">
    <br>
    <button type="submit">Ask</button>
</form>
</div>

<script>
async function uploadText() {
    const text = document.getElementById('upload-text').value;
    if (!text.trim()) {
        alert('Please enter text to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('text', text);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            document.getElementById('upload-status').innerText = '✅ Text stored in vector DB';
            document.getElementById('upload-text').value = ''; // optional: clear text area
        } else {
            document.getElementById('upload-status').innerText = '❌ Upload failed';
        }
    } catch (err) {
        document.getElementById('upload-status').innerText = '❌ Error: ' + err;
    }
}
</script>

</body>
</html>
"""


# Upload endpoint

@app.post("/upload")
def upload(text: str = Form(...)):
    ingest(text)
    return {"status": "success"}



# Ask endpoint

@app.post("/ask", response_class=HTMLResponse)
def ask(question: str = Form(...)):
    start_time = time.time()
    docs = retrieve(question)
    reranked_docs = rerank(question, docs)

    if not reranked_docs:
        return "<h3>No relevant context found.</h3><a href='/'>Back</a>"

    
    context_text = "\n\n".join([d["text"] for d in reranked_docs])
    ans = answer(question, context_text)
    elapsed = round(time.time() - start_time, 2)


    html = f"""
    <h2>Answer</h2>
    <div class="answer-box" style="white-space: pre-wrap;">{ans}</div>


    <h3>Sources</h3>
    """
    for i, d in enumerate(reranked_docs):
        html += f"""
        <div class="source">
        [{i+1}] {d['metadata'].get('source', 'unknown')} | position {d['metadata'].get('position')}
        </div>
        """

    html += f"""
    <p><b>Time taken:</b> {elapsed} seconds</p>
    <a href="/">Ask another question</a>
    """
    return html
