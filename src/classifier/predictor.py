"""
Run a rest API exposing the classification model
"""
import argparse
import os, io
import numpy as np
import json

import torch
import flask
import logging
import log as l
from PIL import Image
from torchvision import transforms

from utils.model import load_trained_model, model_to_tta_model
from utils.data import resize, save_model_input_to_disk

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
top_k = 5
test_time_augmentation = False 

print("predictor cuda", device)
print("return top k of:", top_k)
print("test time augmentation:", test_time_augmentation)
debug = False
prefix = "/opt/ml/"

model_path = os.path.join(prefix, "model", "model.pt")
model = load_trained_model(model_path, device, 40)

if test_time_augmentation:
    model = model_to_tta_model(model)

print("model loaded successful")
labels_path = os.path.join(prefix, "model", "labels.txt")
try:
    with open(labels_path) as f:
        labels = f.read().splitlines()
except:
    print("labels file not found, will not return class name")
    labels = []
img_size = 384


class PredictionService(object):
    #model = None  # Where we keep the model when it's loaded
    transform = None #Image preprocessing

    @classmethod
    def preprocess_img(cls, img):
        """load transforms if not loaded and prep the image for inference"""
        if cls.transform == None:
            cls.transform = transforms.Compose(
            [transforms.ToTensor()])
        img = resize(img, img_size, img_size)     
        img = cls.transform(img).to(device)
        return img[None, ...]


    @classmethod
    def predict(cls, model, img, labels):
        img = cls.preprocess_img(img)

        if debug:
            print("model:", next(model.parameters()).device)
            print("img:", img.device)
        
            save_model_input_to_disk(img)
        outputs = model(img)[0, ...]
        conf = torch.softmax(outputs, 0)
        ordered_conf, ordered_class_id = torch.topk(conf, top_k)
        ordered_conf = ordered_conf.detach().cpu().numpy().tolist()
        ordered_class_id = ordered_class_id.detach().cpu().numpy().tolist()
        
        class_names = []
        for class_id in range(len(ordered_class_id)):
            try:
                class_names.append(labels[class_id])
            except:
                class_names.append("error mapping class_id to a name")
        results = {"classes":ordered_class_id, "confidences": ordered_conf, 'names':class_names}
        json_result = json.dumps(results)
        return json_result

app = flask.Flask(__name__)
logging.raiseExceptions = False
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
LOG_URL = os.getenv('LOG_URL')
if LOG_URL != None:
    log_service_url = LOG_URL
    log_formatter = l.LogServiceFormatter()
    log_handler = l.LogServiceHandler(url=log_service_url)
    log_handler.setFormatter(log_formatter)
    app.logger.addHandler(log_handler)
    app.logger.setLevel(logging.INFO)#TODO: Consider conditional set to debug, based on environment (local dev, prod, etc.)
    app.logger.info('Starting classifier service')

def get_prediction(request):
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))

        predictions = PredictionService.predict(model, img, labels)
        return predictions

#invocations is naming convention expected by Sagemaker
@app.route("/invocations", methods=["POST"])
def invocations():
    if not flask.request.method == "POST":
        return
    return get_prediction(flask.request)

@app.route("/predict", methods=["POST"])
def predict():
    if not flask.request.method == "POST":
        return
    return get_prediction(flask.request)

@app.route("/ping", methods=["GET"])
def ping():
    #only successful if model loads
    health = model is not None  # You can insert a health check here
    print('Model health: ', health)
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing classification model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--model_path", default='', type=str, help="path to model file")
    args = parser.parse_args()
    port = args.port
    #NOTE this will reset default model path (used for testing in virtula environment)
    if len(args.model_path) > 0:
        model_path = args.model_path
    print('port: ', port)
    print('model_path: ', model_path)

    model = torch.hub.load('.', 'custom', model_path, source = 'local')
    app.run(host="0.0.0.0", port=port)  # debug=True causes Restarting with stat
