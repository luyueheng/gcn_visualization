from flask import Flask, request, jsonify, render_template
from json import loads, dumps
import base64
# import numpy as np
from gcn_builder import build_model
import tensorflow as tf

# Hyper Parameters - To Modify
batch_size = 1
lr = 1e-4

model_gcn = build_model(batch_size=1, lr=1e-4)
model_gcn.load_weights('../model/gcn')
print(model_gcn.summary())


# TODO: image prediction

# print('Loading GAN model')
# gan_model = torch.load(filepath)
# gan_model.eval()
# print('GAN model loaded')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def draw():
    return render_template('gcn.html')

@app.route('/generate', methods=['POST'])
def prediction_payload():
    
    print(request.json)
    img_data = None
    with open('static/blank_fake.png', 'rb') as f:
      img_data = base64.b64encode(f.read())
    return jsonify({
      "output": "data:image/png;base64," + img_data.decode()
    })

    # latent_vec = np.array(content['params'])
    # latent_vec = torch.Tensor(latent_vec)
    # latent_vec = latent_vec.view(1, -1, 1, 1, 1)
    # fake = gan_model(latent_vec)
    # np_fake = fake.detach().numpy()
    # voxels = np.reshape(np_fake, (64, 64, 64))
    # voxels = downsample(voxels, 2, method='max')   
    
    # payload = dumps({'voxels':voxels.tolist()})
    # return jsonify(payload)
