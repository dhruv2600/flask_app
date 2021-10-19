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



from reader import answer_question

def get_answer_trial(questionss,ind):

    #return ind when you wanna return a dummy answer
    print(questionss)
    print(ind)
    
    doc_store = ElasticsearchDocumentStore(
        host='localhost',
        username='', password='',
        index=ind,
        )
    
    retriever = TfidfRetriever(document_store=doc_store)
    

    docs=retriever.retrieve(questionss)
    #print(docs)
    print("----------------->")
    print(docs)
    print(docs[0])
    print("--------------->")
    print(docs[0].text)
    print("---------------->")
    print(type(docs[0]))
    print("-------------->")
    answer=answer_question(questionss,docs[2].text)        
    # predict n answers
    return answer

