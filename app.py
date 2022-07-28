from flask import Flask, render_template, flash, redirect, request
from flask.helpers import url_for
from backend import fetchLinkedInAbout, tokeniser, analyze, parsePDF, analyzePDF
import pickle
import spacy

dummy_fn = lambda x:x

nlp = spacy.load("en_core_web_sm")

model_save_location = "./models/"

with open(model_save_location + 'cv.pickle', 'rb') as f:
    cv = pickle.load(f)
with open(model_save_location + 'idf_transformer.pickle', 'rb') as f:
    idf_transformer = pickle.load(f)
with open(model_save_location + 'LR_clf_IE_kaggle.pickle', 'rb') as f:
    lr_ie = pickle.load(f)
with open(model_save_location + 'LR_clf_JP_kaggle.pickle', 'rb') as f:
    lr_jp = pickle.load(f)
with open(model_save_location + 'LR_clf_NS_kaggle.pickle', 'rb') as f:
    lr_ns = pickle.load(f)
with open(model_save_location + 'LR_clf_TF_kaggle.pickle', 'rb') as f:
    lr_tf = pickle.load(f)


app = Flask(__name__)



@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')

@app.route("/resume", methods=['GET','POST'])
def resume():
    if request.method == 'POST':
        filepath = request.form['pathinput']
        result = analyzePDF(filepath)
        return render_template('result.html', result=result)
    return render_template('resume.html')

@app.route("/linkedin", methods=['GET','POST'])
def linkedinparser():
    if request.method == 'POST':
        uname = request.form['profileinput']
        about = fetchLinkedInAbout(uname)
        result = analyze(about)
        return render_template('result.html', result=result)
    return render_template('linkedin.html')


if __name__ == "__main__":
    app.run()
