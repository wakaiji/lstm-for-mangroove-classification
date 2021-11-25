from flask import Flask, flash, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
import pickle
import sys,os,itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from collections import Counter

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo


ALLOWED_EXTENSIONS = set(['xlsx','csv'])

secret_key = os.urandom(12)

DETECT_FOLDER = 'static'
app = Flask(__name__)
app.secret_key = secret_key
app.config['UPLOAD_FOLDER'] = DETECT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

def prediction_mangroove(file_path):
    splitfile = file_path.split(".")
    if splitfile[-1] == "xlsx":
        data_raw = pd.read_excel(file_path)
    else:
        data_raw = pd.read_csv(file_path)

    data_raw.columns = ["ketinggianAir", "suhuAir", "suhuUdara", "kelembapanUdara", "tds", "orp", "do", "ph", "label"]
    X = data_raw.iloc[:,0:9]
    X = to_categorical(X)
    print(X[0])
    load = load_model("model.h5")
    y_predicted = load.predict(X)
    predicted = np.argmax(y_predicted,axis=1)

    target_names = ['Seaward Zone', 'Mid Zone', 'Landward Zone']
    print(predicted)
    loc_pred = count_predict(predicted)

    new_predict = pd.DataFrame(data={'lokasi': target_names, 'total':loc_pred})
    new_predict['precentage'] = (new_predict['total'] / new_predict['total'].sum()) * 100

    visualize_class(new_predict['precentage'].to_list())

    return predicted, target_names, new_predict['precentage'].to_list()

def training_mangroove(file_path):
    splitfile = file_path.split(".")
    if splitfile[-1] == "xlsx":
        data_raw = pd.read_excel(file_path)
    else:
        data_raw = pd.read_csv(file_path)

    data_raw.columns = ["ketinggianAir", "suhuAir", "suhuUdara", "kelembapanUdara", "tds", "orp", "do", "ph", "label", "lokasi"]
    X = data_raw.iloc[:,0:9]
    y = data_raw.iloc[:,-1]
    X = to_categorical(X)
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_train.shape, X_test.shape, y_train.shape, y_test.shape    

    model = Sequential()
    model.add(LSTM(100, input_shape=(9, 720)))
    model.add(Dense(4, activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=["accuracy"])
    model.summary()
    history = model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),verbose=1)
    testing = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    actual = np.argmax(y_test,axis=1)
    predicted = np.argmax(y_pred,axis=1)
    cm = confusion_matrix(actual, predicted)
    visualize_loss(history)
    visualize_accuracy(history)
    target_names = ['Seaward Zone', 'Mid Zone', 'Landward Zone']
    plot_confusion_matrix(cm,target_names,normalize=True)

    return testing[0], testing[1]

def count_predict(predicted):
    one = 0
    two = 0
    three = 0

    for val in predicted:
        if val == 1:
            one+=1
        elif val == 2:
            two+=1
        elif val == 3:
            three+=1
    
    return [one, two, three]

def save_confusion(cm):
    plt.subplots(figsize=(15,8))
    sns.heatmap(data=cm, center=0, annot=True)
    plt.savefig("./static/plot/{}".format("confusion_matrix.png"))
    plt.clf()

def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    #Add Normalization Option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./static/plot/{}".format("confussion.png"))
    plt.clf()

def visualize_loss(history):
    # save image accuracy loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./static/plot/{}".format("loss.png"))
    plt.clf()

def visualize_accuracy(history):
    # save image accuracy model
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./static/plot/{}".format("accuracy.png"))
    plt.clf()


def visualize_class(data_persentasi):
    label = ['Seaward Zone','Mid Zone', 'Landward Zone']

    plt.figure(figsize=(12,7))
    plt.bar(label, data_persentasi)

    plt.title('Persentasi pembagian wilayah', size=16)
    plt.ylabel('Persentasi', size=14)
    plt.xticks(size=12)
    plt.yticks(size=12)

    plt.savefig("./static/plot/{}".format("plotbar.png"))
    plt.clf()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/training")
def training():
    return render_template("training.html")

@app.route("/training", methods=['POST'])
def upload_training_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_folder = os.path.join('static','file')
            file.save(os.path.join(save_folder, filename))
            # flash('file successfully uploaded')

            return redirect(url_for('training_result', filename=filename))
    else:
        flash('Allowed file types are excel')
        return redirect(request.url)

@app.route("/training_result/<filename>")
def training_result(filename):
    file_path = "./static/file/"
    excel_path = file_path + filename
    loss, accuracy = training_mangroove(excel_path)
    return render_template("training_result.html", loss = loss, accuracy = accuracy)

@app.route("/testing")
def testing():
    return render_template("testing.html")

@app.route("/testing", methods=['POST'])
def upload_testing_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_folder = os.path.join('static','file')
            file.save(os.path.join(save_folder, filename))
            # flash('file successfully uploaded')

            return redirect(url_for('testing_result', filename=filename))
    else:
        flash('Allowed file types are excel')
        return redirect(request.url)

@app.route("/testing_result/<filename>")
def testing_result(filename):
    file_path = "./static/file/"
    excel_path = file_path + filename
    prediction_result, target_names, percentage = prediction_mangroove(excel_path)
    return render_template("testing_result.html", prediction = prediction_result, target_names = target_names, percentage = percentage)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
