from flask import Flask, render_template, request, redirect, url_for
import os
from agent.runner import run

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB upload limit

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
