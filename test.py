from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


cwd = os.getcwd()
output_model_path = os.path.join(cwd,"admission_model.pkl")
output_model_path_1 = os.path.join(cwd,"admission_model_1.pkl")
output_model_path_2 = os.path.join(cwd,"admission_predictor.pkl")
output_model_path_3 = os.path.join(cwd,"admission_predictor_new.pkl")
scaler = os.path.join(cwd,"scaler.sav")
output_max_list_path = os.path.join(cwd,"max_list.dat")
output_min_list_path = os.path.join(cwd,"min_list.dat")
scaler = StandardScaler()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict-graduation-admission',methods=["POST"])
# def predict_api():
#     data = request.get_json()
#     df = pd.DataFrame.from_dict(data,orient="index").T

#     df["University Rating"] = df["University Rating"].astype("category")
#     df["Reasearch"] = df["Research"].astype("category")

#     model = pickle.load(open(output_model_path_3,"rb"))
#     max_list = pickle.load(open(output_max_list_path,"rb"))
#     min_list = pickle.load(open(output_min_list_path,"rb"))

#     numerical_columns = ["GRE Score","TOEFL Score","SOP","LOR","CGPA"]

#     for max,min,actual in zip(max_list,min_list,numerical_columns):
#         df[actual] = np.abs((min-df[actual])/(max-df[actual]))

#     prediction = model.predict(df)
#     pre = prediction[0]
#     final_prediction = None
#     if(pre>1.0):
#         final_prediction = 1.0
#     else:
#         final_prediction = pre
#     return str("The probability of the candidate getting admission is {}%".format(np.round(final_prediction*100.0,2)))


@app.route('/predict-graduation-admission-frontend',methods=["POST"])
def predict_front_end():
    data = request.form.to_dict()
    df = pd.DataFrame.from_dict(data,orient="index").T

    print(df.head())
    df["university_rating"] = df["university_rating"].astype("category")
    df["reasearch"] = df["research"].astype("category")

    model = pickle.load(open(output_model_path_3,"rb"))
    max_list = pickle.load(open(output_max_list_path,"rb"))
    min_list = pickle.load(open(output_min_list_path,"rb"))

    numerical_columns = ["gre","toefl","sop","lor","cgpa"]

    for max,min,actual in zip(max_list,min_list,numerical_columns):
        df[actual] = df[actual].astype(float)
        df[actual] = np.abs((min-df[actual])/(max-df[actual]))

    prediction = model.predict(df)
    #print(prediction)
    pre = prediction[0]
    final_prediction = None
    if(pre>1.0):
        final_prediction = 1.0
    else:
        final_prediction = pre
    result = "The probability of the candidate getting admission is {}%".format(np.round(final_prediction*100.0,2))
    return render_template("index.html",prediction_text=result)
    #return str("The probability of the candidate getting admission is {}%".format(np.round(final_prediction*100.0,2)))


if __name__ == "__main__":
    app.run(debug=True)
