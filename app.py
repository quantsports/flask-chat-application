from flask import Flask, request, render_template, jsonify
import api

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        answer = api.get_response(prompt)
        print(answer, 'answeeeer')
        return jsonify({
            'answer': answer,
            'img_path': '',
        })

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)