from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    images = []
    for i in range(1, 7):
        images.append({
            'noisy': f'images/{i}_noisy.png',
            'denoised': f'images/{i}_denoised.png',
            'clean': f'images/{i}_clean.png'
        })
    return render_template('index.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)
