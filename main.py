from flask import Flask
from flask import jsonify, request
from flask_cors import CORS, cross_origin

import transformers
import numpy as np
import tensorflow as tf
import re
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

def bert_encode(data, maximum_length):
  input_ids = []
  attention_masks = []
  #token_type_ids = []

  for i in data:
    text = re.compile('[ㄱ-ㅎ가-힣a-zA-Z0-9]+').findall(i)
    text = " ".join(text)
    encoded = tokenizer.encode_plus(text,
                                    add_special_tokens=True,
                                    max_length=maximum_length,
                                    pad_to_max_length=True, truncation=True)
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])
    #token_type_ids.append(encoded['token_type_ids'])

  return np.array(input_ids), np.array(attention_masks)

loaded_model = tf.keras.models.load_model('bert_model_v2.h5', custom_objects={"TFBertModel": transformers.TFBertModel})


#######################################################################

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def index():
   return "This is youtube-title-predictor api"

@app.route('/predict', methods = ["GET", "POST"])
def predict():
  data = {"requested": "request"}

  params = request.json # getting parameters
  sample_title = params["input"] # extracting sample title

  sample_input_id, sample_attention_mask = bert_encode([sample_title], 30)
  predict_sample = loaded_model.predict([sample_input_id, sample_attention_mask])
  score = str(predict_sample[0][0])

  data["score"] = score
  return jsonify(data)

if __name__ == '__main__':
    app.run()
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)