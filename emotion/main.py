from flask import Flask, render_template
from flask import request
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import flask
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_url_path='/static')


f = open('model.pkl', 'rb')
model = pickle.load(f)  
graph = tf.get_default_graph()

g = open('process.pkl', 'rb')
pro = pickle.load(g)  


@app.route('/')
def home():
	return render_template("home.html")

#defining a /hello route for only post requests
@app.route('/hello', methods=['POST'])
def index():
    #grabs the data tagged as 'name'
    name = request.get_json()['name']
    
    #sending a hello back to the requester
    return "Hello " + name
	
@app.route('/predict', methods=['POST'])
def predict():

	
    #grabbing a set of wine features from the request's body
	feature_array = request.form['feature_array']
	print(feature_array)
	feature_array=[feature_array]
	new1=pro.texts_to_sequences(feature_array)
	new2=pad_sequences(new1, maxlen = 70)
    #our model rates the wine based on the input array
	global graph
	with graph.as_default():
		prediction = model.predict(new2)
	print(prediction)
	prediction1=(list(np.round(np.argmax(prediction, axis=1)).astype(int)))
	prediction2=np.argmax(prediction, axis=1)
	prediction2=prediction2[0]
    
    #preparing a response object and storing the model's predictions
	#response = {}
	#response['predictions'] = prediction
	map=['boredom','happiness','anger','love','sadness','surprise']
    
    #sending our response object back as json
	#return "<h2> Emotion : </h2>" + str(prediction1) + " <h2> Mapping: 'boredom': 0, 'happiness': 1, 'hate': 2, 'love': 3, 'sadness': 4, 'surprise': 5 </h2>"
	return render_template('predict.html', title=map[prediction2], text=feature_array)
if __name__ == '__main__':
    app.run(debug=True)