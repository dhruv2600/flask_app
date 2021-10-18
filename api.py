import flask
from flask_mongoengine import MongoEngine
from flask import render_template, request, redirect
import os
import logging
import json
from flask_cors import CORS
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.farm import FARMReader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from haystack.pipeline import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.utils import print_answers
from haystack.retriever import ElasticsearchRetriever
from haystack.pipeline import ExtractiveQAPipeline
from haystack.retriever.sparse import TfidfRetriever
import docx
import re
from haystack import Finder

from flask_pymongo import PyMongo

from generateans import *



from bson.objectid import ObjectId

# test to insert data to the data base
app = flask.Flask(__name__)
app.run(debug=True)


app.config["MONGO_URI"] = 'mongodb+srv://RithvikS:capstone123@cluster0.m8kbi.mongodb.net/myFirstDatabase?retryWrites=true'
mongo = PyMongo(app)

collection = mongo.db['exams']

papercollection =mongo.db['paper']


@app.route("/test")
def test():
    examid=req.form['examid']
    objInstance = ObjectId(examid)
    c = collection.find({'title': objInstance})
    questionsobj = c.next()['questions']
    length = len(questionsobj)
    dict = {}
    for i in range(length):
        dict[questionsobj[i]['qname']] = 'fill me'

    for key, value in dict.items():
        dict[key] = get_answer_trial(key, value)

    for i in range(length):
        questionsobj[i]['right_answers'] = dict[questionsobj[i]['qname']]

    collection.update_one({'title': "hello!!"}, {
        "$set": {"questions": questionsobj}
    })

    return "Answers updated in MongoDB for the exam"


def computation(right_answer,question,full_marks,student_answer):
    return full_marks

def grade_a_paper(paperID,f_m,r_a,q_a):
    answerslist = papercollection.find_one({'_id': paperID})['answers']
    student_answers=[]
    marks_assigned=[]
    for i in answerslist:
        student_answers.append(i['sans'])
        marks_assigned.append(i['marks_assigned'])
    print(student_answers)
    for i in range(0,len(student_answers)):
        given_marks=computation(r_a[i],q_a[i],f_m[i],student_answers[i])
        marks_assigned[i]=given_marks
    
    c = papercollection.find_one({'_id': paperID})
    answersobj=c['answers']

    for i in range(0,len(answersobj)):
        answersobj[i]['marks_assigned']=marks_assigned[i]    

    papercollection.update_one({'_id':paperID},{
        "$set":{"answers":answersobj,
        "marks":sum(marks_assigned)}
    })
    
    print(marks_assigned)

@app.route("/testgrader")
def testgrader():
    reqid = request.form['examid']
    objInstance = ObjectId(reqid)
    c = collection.find_one({'_id': objInstance})
    student_list = c['student_answers']
    questionsobj = c['questions']
    f_m = []
    r_a = []
    q_a = []
    for i in questionsobj:
        f_m.append(i['fullmarks'])
        r_a.append(i['right_answers'])
        q_a.append(i['qname'])
    print('------------------------->grade_paper')
    for i in student_list:
        grade_a_paper(i,f_m,r_a,q_a)
        print('-------------------------->next paper')

    
    return "all done for now"


@app.route("/test/post", methods=['POST'])
def insert_document():
    req_data = request.get_json()
    collection.insert_one(req_data).inserted_id
    return ('', 204)


@app.route('/test/get')
def get():
    documents = collection.find()
    response = []
    for document in documents:
        document['_id'] = str(document['_id'])
        response.append(document)
    return json.dumps(response)


CORS(app)


@app.route('/update_document', methods=['POST'])
def update_document():
    """Return a the url of the index document."""
    if request.files:
        # index is the target document where queries need to sent.
        index = request.form['index']
        # uploaded document for target source
        train = request.files["doc"]

        doc = docx.Document(train)
        paras = [p.text for p in doc.paragraphs if p.text]
        test_list = []
        for i in range(len(test_list)):
            test_list[i] = re.sub('\t', '', test_list[i])
        data_json = [
            {
                'text': paragraph,
                'meta': {
                    'source': 'essays'
                }
            } for paragraph in test_list
        ]
        doc_store = ElasticsearchDocumentStore(
            host='localhost',
            username='', password='',
            index=index
        )
        doc_store.write_documents(data_json)


@app.route('/get_ans', methods=['POST'])
def get_ans():
    """Return the n answers."""

    question = request.form['question']
    # index is the target document where queries need to sent.
    index = request.form['index']
    print(question)
    print(index)

    doc_store = ElasticsearchDocumentStore(
        host='localhost',
        username='', password='',
        index=index
    )

    retriever = TfidfRetriever(document_store=doc_store)

    print("docs retrieved")

    # initialization of the Haystack Elasticsearch document storage

    # using pretrain model
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",
                        use_gpu=True, no_ans_boost=-10, context_window_size=500,
                        top_k_per_candidate=3, top_k_per_sample=1,
                        num_processes=8, max_seq_len=256, doc_stride=128)

    print("reader initialized successfully")

    # initialization of ElasticRetriever

    # Finder sticks together reader and retriever
    # in a pipeline to answer our actual questions.
    finder = Finder(reader, retriever)

    print("pipeline generated  successfully")

    prediction = finder.get_answers(
        question=question, top_k_retriever=5, top_k_reader=5)
    answer = []
    for res in prediction['answers']:
        answer.append(res['answer'])

    # predict n answers
    print("answers generated successfully")
    return render_template('index.html')


@app.route('/eval_ans', methods=['POST'])
def dummyoper():
    print("post request successful")
    exam_name = request.form["exam-name"]
    c = collection.find_one({"name": exam_name})
    arr = c.questions

    print(exam_name)
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/evaluate', methods=['GET', 'POST'])
def eval():
    return render_template('evaluate.html')


@app.route('/generateanswer', methods=['GET', 'POST'])
def gen():
    return render_template('generateanswer.html')


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return json.dumps({'status': 'failed', 'message':
                       """An internal error occurred: <pre>{}</pre>See logs for full stacktrace.""".format(
                           e),
                       'result': []})


app.run(debug=True, use_reloader=False)
