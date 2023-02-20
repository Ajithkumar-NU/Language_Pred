from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import feature_extraction
from sklearn.linear_model import LogisticRegression
import gc


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


app = Flask(__name__)
model = pickle.load(open('modelUpdated.pkl', 'rb'))

gc.collect()

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
   gc.collect()
   names_gen=pd.read_csv(r"dataset1.csv")
   a = names_gen.sample(frac = 1,random_state=42)

   b = a.dropna(how='all').dropna(how='all',axis=1)
   df2 = b.drop(['Sno', 'Middle Name', 'Language Main'], axis=1)

   # Get the data from the POST request.
   fname = request.form.get('fname')
   sname = request.form.get('sname')

   print("Name is "+fname+" "+sname)

   # Make DataFrame for model
   df2 = df2.append(pd.DataFrame([[fname, sname]], index=[0], columns=df2.columns))
   df2.reset_index(drop=True, inplace=True)

   print(df2.tail)

   # print(df2)
   
   # Get dummies for the categorical variables
   d_f = pd.get_dummies(df2, columns=['First Name', 'Surname'], drop_first=True)

   gc.collect()

   d_f = d_f.iloc[-1, :]
   d_f = d_f.to_frame()
   print(d_f)
   print(d_f.shape)

   # Get the new columns
   new_cols = [col for col in d_f.columns if col not in ['First Name', 'Surname']]

   gc.collect()

   if(len(new_cols) < 4053):
      for i in range(4053-len(new_cols)):
         new_cols.append('dummy'+str(i))
         d_f['dummy'+str(i)] = 0

   gc.collect()

   # Get the predictions
   predictions = model.predict(d_f[new_cols])

   gc.collect()

   print(d_f)
   print(len(predictions))
   print(df2.shape)


   return render_template('results.html', language=predictions[len(predictions)-1], name=df2['First Name'][len(df2)-1]+" "+df2['Surname'][len(df2)-1])

if __name__ == '__main__':
   app.run(debug= True)