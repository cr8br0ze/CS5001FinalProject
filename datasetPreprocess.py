"""
    4/19/2023
    CS5001
    Zichong Meng
    This program file is for Final Project 1600000 twitter data preprocess
"""

#imports
import pandas as pd
from sklearn.model_selection import train_test_split
import math

class twitterData:

    #split data
    def __init__(self, path):
        self.dataset = self.preprocess(path)
        self.trainText,\
            self.valText,\
            self.testText,\
            self.trainScore,\
            self.valScore,\
            self.testScore\
            =self.split(self.dataset)
        self.trainMax = self.maxLen(self.trainText)

    #preprocess data
    def preprocess(self, path):
        dataset = pd.read_csv(path, encoding="latin", header = None)
        dataset = dataset[[0,5]]
        dataset = dataset.rename({0:"score", 5:"text"},axis=1)
        dataset.loc[dataset["score"] == 4, "score"] = 1
        return dataset

    def split(self, dataset):
        cut=math.floor(int(len(dataset))*0.8)
        trainset = dataset.iloc[:cut]
        testset = dataset.iloc[cut:]
        trainText, valText, trainScore, valScore = train_test_split(trainset.text.values, trainset.score.values, train_size=0.8, random_state=42)
        testText, testScore = testset.text.values, testset.score.values
        return trainText, valText, testText, trainScore, valScore, testScore

    def maxLen(self, matrix):
        list = [len(i.split()) for i in matrix]
        return max(list)
