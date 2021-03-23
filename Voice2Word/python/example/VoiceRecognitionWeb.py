from flask import Flask
from flask_cors import CORS

from Demo.Voice2Word.python.example.VoiceRecognition import VoiceRecognition
import requests
app = Flask(__name__)
CORS(app)  # 解决跨域问题


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/train')
def train():
    return 'train success!!!'

@app.route("/voice/<fileName>")
def question_for_web(fileName):
    # 初始化对象
    voiceModel = VoiceRecognition(fileName)

    #组建数据，发送post请求
    result = voiceModel.getRealData()
    post_data = {"data":result}
    svmResult = requests.post("http://localhost:5005/svm/getResult",data=post_data)

    print("result:",result,"    类别:",svmResult.text)
    return svmResult.text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)