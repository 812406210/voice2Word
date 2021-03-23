from flask import Flask,request
from flask_cors import CORS
from Demo.CNTest import RestPython
import jieba
app = Flask(__name__)
CORS(app)  # 解决跨域问题
mymodel= RestPython.RestPython()

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/train')
def train():
    return 'train success!!!'

@app.route("/svm/<user_question>")
def question_for_web(user_question):
    jieba.load_userdict("E:\\python\\Demo\\dict.txt")
    seg_list = jieba.cut(user_question, cut_all=False)
    print("Full Mode: " + ",".join(seg_list))

    label_name_map = ['侮辱性言语', '正常言语']  # 0 1
    result = mymodel.classify(user_question)
    print("result:",label_name_map[int(result)],result)
    return label_name_map[int(result)]

@app.route("/svm/getResult",methods=["POST"])
def getResult():
    label_name_map = ['侮辱性言语', '正常言语']  # 0 1
    if request.method == 'POST':
        data = request.form['data']
        result = mymodel.classify(data)
        print("result:",label_name_map[int(result)],result)
        return label_name_map[int(result)]
    else:
        return "please use POST method"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)