import re
import json
import nltk
import string
import pywapi
import re, math
import datefinder
import datetime, time
import random, operator
import pandas as pd

from dateutil import parser
from itertools import chain
from collections import Counter
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from urllib.request import urlopen
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import NaiveBayesClassifier as nbc
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import sent_tokenize, word_tokenize

class WeatherBot:
    
    def __init__(self):
        self.file = json.loads(open('bot_Q&A.json').read())
        self.intent = None
        self.location = None

    def get_day_of_week(self, question):

        question = question.lower()
        matches = list(datefinder.find_dates(question))

        if len(matches) > 0:
            date = matches[0]
            day = date.strftime('%A')
            
        else:
            if 'today' in question:
                day = ((datetime.date.today()).strftime('%A'))
                
            elif 'day after tomorrow' in question:
                day = ((datetime.date.today() + datetime.timedelta(days = 2)).strftime('%A'))
                
            elif 'tomorrow' in question:
                day = ((datetime.date.today() + datetime.timedelta(days = 1)).strftime('%A'))
                
            else:
                day = ((datetime.date.today()).strftime('%A'))
                
        return day

    def get_location(self, question):

        location_tagger = StanfordNERTagger('Tagger/stanford-ner-2016-10-31/classifiers/english.conll.4class.distsim.crf.ser.gz', 'Tagger/stanford-ner-2016-10-31/stanford-ner-3.7.0.jar')
        question = question.title()
        tag = location_tagger.tag(question.split())
        loc_word = ''
        for word,tag in tag:
            if(tag == 'LOCATION'):
                loc_word = loc_word + ", " + word
            loc_word = loc_word.strip()
        if loc_word == '':
            loc_word = None
        return loc_word

    def get_location_id(self, city):
         # Get location ID for that city
        count = 0
        flag = True
        lookup = pywapi.get_location_ids(city)
        while len(lookup) != 1:
            if len(lookup) > 1:
                for key,value in lookup.items():
                    if 'India' in value:
                        location_id = key
                        city = value
                        lookup = pywapi.get_location_ids(city)
                        flag = False
                        break
            if flag == False:
                break
            if len(lookup) == 0:
                return "Fail",None
        for k in lookup:
            location_id = k
        return location_id,city

    def get_weather_of_day(self, weather_com, day_of_week):
        weekday = ''
        for get_day in range(len(weather_com['forecasts'])):
            if day_of_week == weather_com['forecasts'][get_day]['day_of_week']:
                weekday = get_day
                break
        return weekday

    def get_traindata(self):
        train_csv = pd.read_csv('weathertrain.csv',header= None,names = ['sentence','label'])
        subset = train_csv[['sentence', 'label']]
        tuples = [tuple(x) for x in subset.values]
        return tuples

    def trainNBC(self):
        data = self.get_traindata()
        stop_words = set(stopwords.words("english"))
        d1 = [(' '.join(list((i for i in word_tokenize(sentence) if i not in stop_words))),tag) for sentence, tag in data ]
        vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in d1 if i not in stop_words]))
        feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in d1]
        classifier = nbc.train(feature_set)
        return vocabulary,classifier

    def get_class(self, query):
        test_sentence = query
        vocabulary,classifier = self.trainNBC()
        featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary}
        count = 0
        for k,v in featurized_test_sentence.items():
            if v == True:
                count += 1
            if count == 0:
                ans = None
            else:
                ans = classifier.classify(featurized_test_sentence)
        return ans

    def get_weather(self, city, day_of_week, old_category, query):
        # Getting location id for that city  
        location_id, city = self.get_location_id(city)
        # Getting weather data for that city
        weather_com = pywapi.get_weather_from_weather_com(location_id)    
        # Getting weather data for that day
        weekday = self.get_weather_of_day(weather_com,day_of_week)
        output = {
                  'maxtemp':32,
                  'mintemp':24,
                  'avgtemp':28,
                  'temp':29,
                  'rain':'No',
                  'percentrain_day':'15',
                  'percentrain_night':'15',
                  'weather_day':'Clear',
                  'weather_night':'Clear',
                  'flag':'not rain',
                  'city':city,
                  'category':old_category,
                  'day_of_week':day_of_week
                 }
        if weekday == '':
            return None
        output['temp'] = int(weather_com['current_conditions']['temperature'])
        weekday_weather = weather_com['forecasts'][weekday]
        category = self.get_class(query)
        output['maxtemp'] = weekday_weather['high']
        output['mintemp'] = weekday_weather['low']
        output['avgtemp'] = (int(weekday_weather['high']) + int(weekday_weather['low']))/2
        output['percentrain_day'] = weekday_weather['day']['chance_precip']
        output['percentrain_night'] = weekday_weather['night']['chance_precip']
        if int(output['percentrain_day']) > 30 or  int(output['percentrain_night']) > 30:
            output['rain'] = 'Yes' 
            output['flag'] = 'rain'
        output['weather_day'] = weekday_weather['day']['text'] if weekday_weather['day']['text'] else "Clear"
        output['weather_night'] = weekday_weather['night']['text'] if weekday_weather['night']['text'] else "Clear"
        if category != None:
            output['category'] = category
        return output

    def get_cosine(self, vec1, vec2):
         intersection = set(vec1.keys()) & set(vec2.keys())
         numerator = sum([vec1[x] * vec2[x] for x in intersection])
         sum1 = sum([vec1[x]**2 for x in vec1.keys()])
         sum2 = sum([vec2[x]**2 for x in vec2.keys()])
         denominator = math.sqrt(sum1) * math.sqrt(sum2)

         if not denominator:
            return 0.0
         else:
            return float(numerator) / denominator

    def text_to_vector(self, text):
         WORD = re.compile(r'\w+')
         words = WORD.findall(text)
         return Counter(words)

    def get_question_match(self, user_question, question_list):
        question_similarity = {}
        for i in question_list:
            cosine = self.get_cosine(self.text_to_vector(user_question.lower()), self.text_to_vector(i.lower()))
            question_similarity[i] = cosine
        sorted_questions = sorted(question_similarity.items(), key=operator.itemgetter(1), reverse=True)  
        return sorted_questions[0][0]

    def get_response(self, question, category):
        similar_question = self.get_question_match(question, list(self.file[category].keys()))
        response_to_user = random.choice(self.file[category][similar_question])
        return response_to_user


    def respond(self,query):
        query = ' '.join(l for l in word_tokenize(query) if l not in string.punctuation)
        category = self.get_class(query)
        city = self.get_location(query)
        day_of_week = self.get_day_of_week(query)
        
        if category == None and city == None:
            output = self.get_response(query, "random")
            return output
        
        elif city == None and self.location == None:
            self.intent = category
            return 'Please enter your location'
        
        elif city != None and category == None:
            self.location = city
            return 'What do you wanna Know?'
           
        elif city == None and category != None:
            city = self.location
            weather = self.get_weather(city,day_of_week, category, query)
            if weather == None:
                return 'Cannot predict data for more than 5 days'
            else:
                response_user = self.get_response(query, category)
                output = response_user.format(**weather)
                self.intent = category
                self.location = city
                return output
    
        elif city != None and category == None:
            category = self.intent
            weather = self.get_weather(city, day_of_week, category, query)
            self.location = city
            if weather == None:
                return 'Cannot predict data for more than 5 days'
            else:
                response_user = self.get_response(query, category)
                output = response_user.format(**weather)
                self.intent = category
                self.location = city
                return output
        
        else:
            self.location = city
            self.intent = category
            weather = self.get_weather(city, day_of_week, category, query)
            if weather == None:
                return 'Cannot predict data for more than 5 days'
            else:
                response_user = self.get_response(query, category)
                output = response_user.format(**weather)
                self.intent = category
                self.location = city
                return output    
