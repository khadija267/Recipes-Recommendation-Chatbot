from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.metrics import pairwise_distances_argmin_min
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer 
df = pd.read_pickle("df_final.pkl")
wv = Word2Vec(df['ingredients filtered'], min_count = 1, window = 5)
cv = CountVectorizer()
cv.fit_transform(df['ingredients filtered text']).todense() 


#from flask import jsonify

# import subprocess
# import sys
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])    
# install("gensim")    
# wv = pickle.load(open('wv.pkl', 'rb'))
# wv2 = pickle.load(open('wv2.pkl', 'rb'))

cv_recipe=pickle.load(open('cv_recipes.pkl', 'rb'))
cv_ingre=pickle.load(open('cv_ingredients.pkl', 'rb'))

tfidf = pickle.load(open('tfidf.pkl', 'rb'))
cluster_recipe= pickle.load(open('clustering_model_recipes.pkl', 'rb'))
cluster_ingre= pickle.load(open('clustering_model_inredients.pkl', 'rb'))
classify= pickle.load(open('classification_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/') # this is the home page route
def hello_world(): # this is the home page function that generates the page code
    return "Hello world!"
    
@app.route('/webhook', methods=['POST','GET'])
def webhook():
  sum=0
  req = request.get_json(silent=True, force=True)
  fulfillmentText = ''
  query_result = req.get('queryResult')
  print(query_result.get('action'))
  ##################################3
  if query_result.get('action') == 'meal':
    print("aaa")
    meal = str(query_result.get('parameters').get('meal'))
    res=str(list(df[df['title']==meal]['ingredients']))
    fulfillmentText = 'The ingredients are '+res
  # similar ingredients  
  elif query_result.get('action') == 'similar.recipe':
    print("a")
    ing = np.array((query_result.get('parameters').get('test_ingredients')))
    print("b")
    print(ing)
    s=cv_recipe.transform(ing).todense()
    print("c")
    t = cv_recipe.transform(df['ingredients filtered text']).todense()
    print("fff")
    closest, _ = pairwise_distances_argmin_min(s, t, metric='cosine')
    print(closest)
    res=str(np.array(df[['title','url']])[closest[0]])
    fulfillmentText = 'The most similar meals are '+res  
    #################
  elif query_result.get('action') == 'similar.ingredients':
    print("aa")
    ing = str(query_result.get('parameters').get('test_ingredients'))
  
    x = str([x for x, c in wv.wv.most_similar(ing, topn=10)])
    fulfillmentText = 'I recommend to you the follow ingredients '+ x

  elif query_result.get('action') == 'add.numbers':
    num1 = int(query_result.get('parameters').get('number'))
    num2 = int(query_result.get('parameters').get('number1'))
    sum = str(num1 + num2)
    print('here num1 = {0}'.format(num1))
    print('here num2 = {0}'.format(num2))
    fulfillmentText = 'The sum of the two numbers is '+sum
  return {
        "fulfillmentText": fulfillmentText,
        "source": "webhookdata"
    }
    
   
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080) # This line is required to run Flask on repl.it