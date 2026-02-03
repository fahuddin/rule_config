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
import redis

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB upload limit

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB   = int(os.environ.get("REDIS_DB", "0"))

rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

RULE_FIELDS = ["rule_id", "application", "sub_module", "rule_type", "rule_name", "rule_def", "rule_desc"]
# Simple in-memory cache: key -> (description, timestamp)
_desc_cache = {}
_cache_lock = Lock()
# TTL for cache entries (seconds)
CACHE_TTL = 24 * 3600

def list_rule_id():
    ids = rdb.smembers("rule:all")
    if ids:
        return sorted(ids, key=lambda x: int(x))
    found = []
    for key in rdb.scan_iter(match="rule:*", count=500):
        if key in ("rule:id", "rule:all"):
            continue
        if key.startswith("rule:"):
            rid = key.split(":", 1)[1]
            if rid.isdigit():
                found.append(rid)
    return sorted(set(found), key=lambda x: int(x))

def _get_rule(rule_id: str) -> dict:
    d = rdb.hgetall(f"rule:{rule_id}") or {}
    # normalize: ensure all expected fields exist
    for f in RULE_FIELDS:
        d.setdefault(f, "")
    # ensure rule_id is present
    if not d["rule_id"]:
        d["rule_id"] = str(rule_id)
    return d


def _get_all_rules() -> list[dict]:
    ids = list_rule_id()
    return [_get_rule(rid) for rid in ids]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/api/rules", methods=["GET"])
def api_rules():
    rules = _get_all_rules()
    return jsonify({"count": len(rules), "rules": rules})


@app.route("/api/rules/<rule_id>", methods=["GET"])
def api_rule(rule_id):
    rule = _get_rule(rule_id)
    if not rule or (not rdb.exists(f"rule:{rule_id}")):
        return jsonify({"error": "not found"}), 404
    return jsonify(rule)

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
    "You are an expert at translating rule logic into plain English for non-technical users. "
    "Given the following MVEL rule definition, write a clear, concise description (1–2 sentences) "
    "that explains what the rule checks and what happens when it matches. "
    "Avoid code syntax, variable names, and implementation details. "
    "Focus on the business meaning and outcome.\n\n"
    f"MVEL Rule:\n{definition}\n\n"
    "Plain-English Description:")

    def _clean_text(text: str, definition: str) -> str:
        """Clean generated text by removing assistant lead-ins, extraneous characters,
        and sentences that simply restate the provided definition.
        """
        if not text:
            return ''
        t = text.strip()
        # remove common assistant lead-ins like "Here is..." or "Here's..."
        t = re.sub(r"^\s*(Here is(?: a)?(?: .*?)?:|Here\'s:?|Here you go:?|Here you are:?|Assistant:|Response:)\s*\n*", '', t, flags=re.I)
        # remove any leading label line that ends with ':'
        t = re.sub(r"^.*?:\s*\n+", '', t, count=1)
        # normalize whitespace and remove control chars
        t = re.sub(r"[\x00-\x1f\x7f]+", ' ', t)
        t = re.sub(r"\s+", ' ', t).strip()

        # split into sentences (simple heuristic) include : and ; as sentence boundaries
        sentences = re.split(r'(?<=[\.!?;:])\s+', t)

        def word_set(s: str):
            return set([w for w in re.findall(r"\w+", s.lower())])

        def is_restatement(s: str) -> bool:
            # skip very short fragments
            if len(s.strip()) < 10:
                return True
            # common starter phrases that often repeat the rule
            if re.match(r"^\s*(the rule|this rule|it checks|it validates|it ensures|it verifies|ensures|verifies|validates|checks)\b", s.strip(), flags=re.I):
                return True
            # markers that indicate paraphrase/alternate phrasing
            if re.search(r"\balternatively\b|\balso\b|\bin summary\b|\bfor example\b", s, flags=re.I):
                return True
            # high overlap with definition
            def_words = word_set(definition)
            sent_words = word_set(s)
            if not def_words or not sent_words:
                return False
            overlap = len(def_words & sent_words) / max(1, len(def_words))
            # also consider proportion relative to sentence length
            overlap_sent = len(def_words & sent_words) / max(1, len(sent_words))
            if overlap > 0.45 or overlap_sent > 0.45:
                return True
            # heuristic: phrases that indicate a validation restatement
            if 'validation' in s.lower() and 'account' in s.lower():
                return True
            return False

        kept = []
        for s in sentences:
            if not s or is_restatement(s):
                continue
            kept.append(s.strip())

        out = ' '.join(kept).strip()
        # if aggressive filtering removed everything, fall back to the cleaned original text
        if not out:
            out = t
        # remove quotation characters (straight and smart quotes)
        out = out.strip()
        out = out.replace('"', '').replace("'", '')
        out = out.replace('“', '').replace('”', '').replace('‘', '').replace('’', '')
        out = out.replace('«', '').replace('»', '')
        # final cleanup: remove repeated newlines and normalize whitespace
        out = re.sub(r"[\r\n]+", ' ', out)
        out = re.sub(r"\s+", ' ', out).strip()
        return out

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

    cleaned = _clean_text(text, definition)
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
