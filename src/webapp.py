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

#redis config
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




@app.route('/api/generate-description', methods=['POST'])
def generate_description():
    data = request.get_json(force=True) or {}
    definition = data.get('definition', '')
    model = data.get('model', 'llama3.1')
    mode = data.get('mode', "verify")
    force = bool(data.get('force', False))

    def _clean_text(text: str) -> str:
        if not text:
            return ''
        t = text.strip()
        t = re.sub(r"^\s*(Here is(?: a)?(?: .*?)?:|Here\'s:?|Here you go:?|Here you are:?|Assistant:|Response:)\s*\n*", '', t, flags=re.I)
        t = re.sub(r"^.*?:\s*\n+", '', t, count=1)
        t = re.sub(r"[\x00-\x1f\x7f]+", ' ', t)
        t = re.sub(r"\s+", ' ', t).strip()

        sentences = re.split(r'(?<=[\.!?;:])\s+', t)

        def word_set(s: str):
            return set([w for w in re.findall(r"\w+", s.lower())])

        def is_restatement(s: str) -> bool:
            if len(s.strip()) < 10:
                return True
            if re.match(r"^\s*(the rule|this rule|it checks|it validates|it ensures|it verifies|ensures|verifies|validates|checks)\b", s.strip(), flags=re.I):
                return True
            if re.search(r"\balternatively\b|\balso\b|\bin summary\b|\bfor example\b", s, flags=re.I):
                return True

            def_words = word_set(definition)
            sent_words = word_set(s)
            if not def_words or not sent_words:
                return False

            overlap = len(def_words & sent_words) / max(1, len(def_words))
            overlap_sent = len(def_words & sent_words) / max(1, len(sent_words))
            if overlap > 0.45 or overlap_sent > 0.45:
                return True

            if 'validation' in s.lower() and 'account' in s.lower():
                return True
            return False

        kept = []
        for s in sentences:
            if not s or is_restatement(s):
                continue
            kept.append(s.strip())

        out = ' '.join(kept).strip()
        if not out:
            out = t

        out = out.strip()
        out = out.replace('"', '').replace("'", '')
        out = out.replace('“', '').replace('”', '').replace('‘', '').replace('’', '')
        out = out.replace('«', '').replace('»', '')
        out = re.sub(r"[\r\n]+", ' ', out)
        out = re.sub(r"\s+", ' ', out).strip()
        return out

    # ✅ cache key MUST include mode (and preferably force doesn't cache anyway)
    key = hashlib.sha256((definition + '||' + model + '||' + mode).encode('utf-8')).hexdigest()

    # ✅ Cache read (return BOTH description + trace so UI updates even on cache hits)
    if not force:
        with _cache_lock:
            entry = _desc_cache.get(key)
            if entry:
                payload, ts = entry
                if time.time() - ts < CACHE_TTL:
                    return jsonify(payload), 200
                else:
                    del _desc_cache[key]

    # ✅ Always define result/text/trace
    try:
        result = run(mode=mode, mvel_texts=[definition], model=model, enable_trace=True)
        if isinstance(result, str):
            result = {"output": result, "trace": None}
    except Exception as e:
        logging.exception("Error running agent")
        result = {"output": f"Error running agent: {e}", "trace": None}

    text = (result.get("output") or "").strip()
    trace = result.get("trace")

    cleaned = _clean_text(text)

    payload = {"description": cleaned, "trace": trace}

    if not force:
        try:
            with _cache_lock:
                _desc_cache[key] = (payload, time.time())
        except Exception:
            logging.exception('Failed to write cache')

    return jsonify(payload), 200


@app.route("/api/rules/<rule_id>/description", methods=["POST"])
def api_save_description(rule_id):
    data = request.get_json(force=True) or {}
    desc = (data.get("description") or "").strip()
    if not desc:
        return jsonify({"error": "description required"}), 400

    key = f"rule:{rule_id}"
    if not rdb.exists(key):
        return jsonify({"error": "rule not found"}), 404

            # store as a new field (rule_desc)
    rdb.hset(key, "rule_desc", desc)
    return jsonify({"ok": True, "rule_id": rule_id, "rule_desc": desc})
                

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
