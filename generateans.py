import flask
from flask_mongoengine import MongoEngine
from flask import render_template, request,redirect
import os
import logging
import json
from flask_cors import CORS
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.farm import FARMReader
from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline
from haystack.pipeline import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.utils import print_answers
from haystack.retriever import ElasticsearchRetriever
from haystack.pipeline import ExtractiveQAPipeline
from haystack.retriever.sparse import TfidfRetriever
from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline
import docx
import re
from haystack import Finder

from flask_pymongo import PyMongo

def get_answer_trial(questionss,ind):

    print(questionss)
    print(ind)
    
    doc_store = ElasticsearchDocumentStore(
        host='localhost',
        username='', password='',
        index=ind
        )
    
    retriever = TfidfRetriever(document_store=doc_store)
    
    print("docs retrieved")

    #initialization of the Haystack Elasticsearch document storage


    # using pretrain model
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",
                use_gpu=True, no_ans_boost=-10, context_window_size=500,
                top_k_per_candidate=1, top_k_per_sample=1,
                num_processes=8, max_seq_len=256, doc_stride=128)
    
    print("reader initialized successfully")

    #initialization of ElasticRetriever
    
    # Finder sticks together reader and retriever
    # in a pipeline to answer our actual questions.
    pipeline = ExtractiveQAPipeline(reader, retriever)
    print("pipeline generated  successfully")

    prediction = pipeline.run(query=questionss, top_k_retriever=1, top_k_reader=1)
    answer = []
    for res in prediction['answers']:
        answer.append(res['answer'])
        
    # predict n answers
    return answer


def computation(question,right_answer,student_answer,full_marks):
    return 0
    