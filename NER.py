import tweepy
import pandas as pd
import csv  # Import csv
import xlrd
import codecs
import nltk

#from monkeylearn import MonkeyLearn
#ml = MonkeyLearn('cc10e828cafbfe08d80a7dfd1dff4bfa9968b93b')
#data = ["['අද'  'ආණ්ඩුවට'  'ඩඩ්ලිගෙ'  'හාල්'  'මිලවත්'  'පාලනය'  'කරගන්න'  'බෑ'  '7']"]
#model_id = 'ex_yAXTMBc4'
#result = ml.extractors.extract(model_id, data)
#print(result.body)


# -*- coding: utf-8 -*-

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

#st = StanfordNERTagger('stanford-ner-tagger/english.all.3class.distsim.crf.ser.gz',
#					   'stanford-ner-tagger/stanford-ner.jar',
#					   encoding='utf-8')

#text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

#tokenized_text = word_tokenize(text)
#classified_text = st.tag(tokenized_text)

#print(classified_text)

#**********************************************************************************
import os
os.environ['IBM_JAVA_OPTIONS']="-Dfile.encoding=UTF-8 -Xmx16G"
import nltk
from nltk.tag.stanford import StanfordNERTagger


sentence = "පහළ කාණ්ඩයේ සමස්ත ලංකා පැසිපන්දු තරගාවලියේ ශූරතාව දිනාගැනීමට මෝදර ඩිලාසාල්	විදුහල සමත්විය"

jar = './stanford-ner-tagger/stanford-ner.jar'
model = './stanford-ner-tagger/dummy-ner-model-sinhala.ser.gz'

# Prepare NER tagger with english model
ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')

# Tokenize: Split sentence into words
words = nltk.word_tokenize(sentence)

# Run NER tagger on words
print(ner_tagger.tag(words))

