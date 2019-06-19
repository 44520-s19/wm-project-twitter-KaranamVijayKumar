# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 00:46:53 2019

@author: S534627
"""

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiments(text):
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def split_sentiments(sentiments):
    xs = [sent['neg'] for sent in sentiments]
    ys = [sent['neu'] for sent in sentiments]
    zs = [sent['pos'] for sent in sentiments]
    
    return xs , ys, zs