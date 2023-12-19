from flask import Flask, jsonify, request
import MLmodel

# Flask App
app = Flask(__name__)

# API status result
@app.route('/classify_api', methods=["GET"])
def claasify_api():

    input_text = request.args.get('input_text', '')

    output = MLmodel.classification(input_text)
    accuracy = "{:.2%}".format(MLmodel.accuracy)

    result = {'code': 200,
              'running': True,
              'text': input_text, 
              'prediction': output,
              'accuracy': accuracy}    
    return jsonify(result)
    

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(port=8080, debug=True)
