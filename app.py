from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  
import cohere

app = Flask(__name__)
CORS(app)

co = cohere.Client('fYFuCn6QnU3RwGGkADytE4XjeXwwP9qd92HD5XGK')  

def generate_question(topic):
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=f"Generate a relevant question about {topic}:",
        max_tokens=50,
        temperature=0.5
    )
    return response.generations[0].text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_question', methods=['POST'])
def question():
    topic = request.json.get('topic')
    question = generate_question(topic)
    return jsonify({'question': question})

@app.route('/evaluate_answer', methods=['POST'])
def evaluate():
    question = request.json.get('question')
    user_answer = request.json.get('answer')
    prompt = f"""
    Question: {question}
    User's answer: {user_answer}
    Evaluate this answer based on correctness and relevance:
    """
    
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=40,
        temperature=0.5
    )
    
    evaluation = response.generations[0].text.strip()
    return jsonify({'evaluation': evaluation})

if __name__ == '__main__':
    app.run(debug=True)
