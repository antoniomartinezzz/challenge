from flask import Flask, request, send_file, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
from deploy import recognize_NN, recognize_RF

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER_NN'] = './test_result_NN/'
app.config['PROCESSED_FOLDER_RF'] = './test_result_RF/'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER_NN'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER_RF'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    image = request.files['image']
    method = request.form['method']
    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        if method == 'nearest_neighbor':
            output_image = recognize_NN(image_path)
            output_path = os.path.join(app.config['PROCESSED_FOLDER_NN'], filename)
            
        else:
            output_image = recognize_RF(image_path)
            output_path = os.path.join(app.config['PROCESSED_FOLDER_RF'], filename)

        cv2.imwrite(output_path, output_image)

        return redirect(url_for('display_image', filename=filename, method=method))
    return redirect(url_for('index'))

@app.route('/<method>/<filename>')
def display_image(method, filename):
    if method == 'nearest_neighbor':
        folder = app.config['PROCESSED_FOLDER_NN']
    else:
        folder = app.config['PROCESSED_FOLDER_RF']
    image_url = os.path.join(folder, filename)
    print(image_url)
    return render_template('display_image.html', image_url=image_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
