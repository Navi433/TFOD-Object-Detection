from flask import Flask ,request,jsonify,render_template
import os
from research.obj import MultiClassObj
from com_in_ineuron_ai_utils.utils import decodeImage
from flask_cors import CORS, cross_origin

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = 'inputImage.jpg'
        modelpath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        self.objectDetection = MultiClassObj(self.filename,modelpath)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
@cross_origin()
def predictRoute():
    image= request.json['image']
    decodeImage = ( image, clApp.filename)
    result =clApp.objectDetection.getPrediction()
    return jsonify(result)


if __name__ == '__main__':
    clApp = ClientApp()
    port = 5000
    app.run(host='0.0.0.0', port=port)







