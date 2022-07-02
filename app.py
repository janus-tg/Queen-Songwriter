from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def formInput():
    text = request.form['text']
    print(text)
    return render_template('index.html', oldInput = text, newLyrics = text + "resultwefn urwibhiwufw euiore2or oeirhe20or oehri2oebr oeirhoeb oierhoerbh oehroie2hro oeiufewo")

if __name__ == "__main__":
    app.run(debug=True)