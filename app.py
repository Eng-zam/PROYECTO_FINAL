import numpy as np
# from xgboost import XGBClassifier
# from sklearn.ensemble import XGBoost as xgb
# from xgboost.sklearn import XGBClassifier
import pickle
# from sklearn.externals import joblib       
from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
import pandas as pd
from flask import Flask, request, render_template
import lightgbm
# import joblib


app = Flask(__name__)
model = pickle.load(open('modelo_credito.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]

    features_value = [np.array(input_features)]

    features_name = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL',
                     'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE',
                     'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
                     'MONTHS_BALANCE']
    df = pd.DataFrame(features_value, columns=features_name)
    print(df)
    output = model.predict(df)
        
    if output == 0:
        res_val = "** Pagador **"
    elif output == 1:
        res_val = "** Moroso **"
    return render_template('index.html', prediction_text='Prediccion de credito {}'.format(res_val))

if __name__ == "__main__":
#     app.debug = True
    app.run(host="0.0.0.0", port =80)

