from flask import Flask, request, jsonify, render_template
import base64
from gcn_builder_no_outline import GCN, json_to_data, data_to_image
import tensorflow as tf
import os
import matplotlib.pyplot as plt


# Build Model
model_gcn = GCN('MAE')
# Load Weights
model_gcn.load_weights('../model/gcn_mae50')
print('gcn model loaded')


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('gcn.html')


@app.route('/generate', methods=['POST'])
def generate():
    
    data = json_to_data(request.json)
    pred_data = model_gcn.predict(data)
    img_list = data_to_image(data, pred_data, H=64, W=64, alpha=180)
    
    for img in img_list:
        # Only show the center part of plots, Since they are too small
        # Can remove slicing
        plt.imshow(img[15:-15,15:-15,:])
        plt.axis('Off')
    plt.savefig('predict.png', dpi = 128, transparent=False, bbox_inches='tight' , pad_inches=0)
    plt.clf()
    plt.close('all')


    img_data = None
    with open('predict.png', 'rb') as f:
      img_data = base64.b64encode(f.read())
    os.remove('predict.png')

    return jsonify({
      "output": "data:image/png;base64," + img_data.decode()
    })
