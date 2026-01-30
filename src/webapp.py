from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from agent.runner import run
from agent.llm import get_llm
# Use message helper compatible with installed langchain_core
from langchain_core.messages.human import HumanMessage
import logging
import hashlib
import time
import re
from threading import Lock

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB upload limit

# Simple in-memory cache: key -> (description, timestamp)
_desc_cache = {}
_cache_lock = Lock()
# TTL for cache entries (seconds)
CACHE_TTL = 24 * 3600

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/rule-config', methods=['GET'])
def rule_config():
    return render_template('rule_config.html')

@app.route('/process', methods=['POST'])
def process():
    mode = request.form.get('mode', 'explain')
    model = request.form.get('model', 'llama3.1')
    trace = bool(request.form.get('trace'))

    uploaded = request.files.get('mvelfile')
    if not uploaded or uploaded.filename == '':
        return redirect(url_for('index'))

    text = uploaded.read().decode('utf-8', errors='replace')

    try:
        result = run(mode=mode, mvel_texts=[text], model=model, enable_trace=trace)
    except Exception as e:
        result = f"Error running agent: {e}"

    return render_template('result.html', mode=mode, filename=uploaded.filename, output=result)



# API: generate a short user-facing description from a rule definition using the local LLM
@app.route('/api/generate-description', methods=['POST'])
def generate_description():
    data = request.get_json(force=True) or {}
    definition = data.get('definition', '')
    model = data.get('model', 'llama3.1')
    force = bool(data.get('force', False))

    if not definition:
        return jsonify({'description': ''}), 200

    prompt = (
        "You are a concise rule documentation assistant. "
        "Given the following rule definition, produce a short (1-2 sentence) user-facing description explaining the purpose and effect of the rule:\n\n"
        f"Definition: {definition}\n\nDescription:")

    def _clean_text(text: str) -> str:
        if not text:
            return ''
        t = text.strip()
        # remove common assistant lead-ins like "Here is..." or "Here is a 1-2 sentence..."
        t = re.sub(r"^\s*(Here is(?: a)?(?: .*?)?:|Here\'s:?|Here you go:?|Here you are:?|Assistant:|Response:)\s*\n*", '', t, flags=re.I)
        # remove any leading label line that ends with ':'
        t = re.sub(r"^.*?:\s*\n+", '', t, count=1)
        # strip surrounding quotes
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            t = t[1:-1].strip()
        return t

    # cache key based on normalized definition and model
    key = hashlib.sha256((definition + '||' + model).encode('utf-8')).hexdigest()
    # check cache unless force=true
    if not force:
        with _cache_lock:
            entry = _desc_cache.get(key)
            if entry:
                desc, ts = entry
                if time.time() - ts < CACHE_TTL:
                    return jsonify({'description': desc}), 200
                else:
                    del _desc_cache[key]

    try:
        llm = get_llm(model=model, temperature=0.0)
        # Use HumanMessage to call chat-style generate
        res = llm.generate([[HumanMessage(content=prompt)]])
        text = ''
        gens = getattr(res, 'generations', None)
        if gens and len(gens) and len(gens[0]):
            gen0 = gens[0][0]
            # prefer plain text, then message.content
            if hasattr(gen0, 'text') and gen0.text:
                text = gen0.text
            elif hasattr(gen0, 'message') and getattr(gen0.message, 'content', None):
                text = gen0.message.content
            else:
                text = str(gen0)
    except Exception as e:
        logging.exception('LLM generation failed')
        text = ''

    cleaned = _clean_text(text)
    # store in cache
    try:
        with _cache_lock:
            _desc_cache[key] = (cleaned, time.time())
    except Exception:
        logging.exception('Failed to write cache')

    return jsonify({'description': cleaned}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
