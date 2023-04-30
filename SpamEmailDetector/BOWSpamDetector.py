import os
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import re
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from SpamEmailDetector.Globals.GlobalFunctions import show_metrics
from sklearn import metrics
import seaborn as sns
import joblib

class BOWSpamDetector:
    def __init__(self,normalTextFilePATH:str,spamTextFilePATH:str,stopWordsTextFilePATH:str):
        '''Initialize all files'''
        self.normalTextFilePATH = normalTextFilePATH
        self.spamTextFilePATH = spamTextFilePATH
        self.stopWordsTextFilePATH=stopWordsTextFilePATH
        def readTXTFile(path:str)->list:
            file = open(path)
            lines = file.readlines()
            file.close()
            return lines
        # self.normalEmails=[''.join(re.findall('[\u4e00-\u9fa5]',i)) for i in readTXTFile(self.normalTextFilePATH)]
        # self.spamEmails = [''.join(re.findall('[\u4e00-\u9fa5]',i)) for i in readTXTFile(self.spamTextFilePATH)]
        self.normalEmails = readTXTFile(self.normalTextFilePATH)
        self.spamEmails = readTXTFile(self.spamTextFilePATH)
        self.stopWords = [i.strip() for i in readTXTFile(self.stopWordsTextFilePATH)]

    def preprocessEmails(self,ChineseOnly:bool)->None:
        if ChineseOnly==True:
            # 只保留中文
            self.normalEmails = [''.join(re.findall('[\u4e00-\u9fa5]', i)) for i in self.normalEmails]
            self.spamEmails = [''.join(re.findall('[\u4e00-\u9fa5]', i)) for i in self.spamEmails]
        else:
            # 首先剔除url，其次保留中英文
            def removeURLRetainChineseEnglish(string):
                removeURLResult = re.sub(r"[a-zA-z]+://[^s]*", '', string)
                return ''.join(re.findall(r"[\u4e00-\u9fa5a-zA-Z]", removeURLResult))
            self.normalEmails = [removeURLRetainChineseEnglish(i) for i in self.normalEmails]
            self.spamEmails = [removeURLRetainChineseEnglish(i) for i in self.spamEmails]
        return None

    def createVector(self,randomSeed=2020310221,tagAsModelofClass=False):
        if tagAsModelofClass==True:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.normalEmails+self.spamEmails,\
                                                                [1]*len(self.normalEmails)+[0]*len(self.spamEmails),\
                                                                test_size = 0.33,\
                                                                random_state = randomSeed)
            countVectorizer = CountVectorizer(tokenizer=jieba.lcut,stop_words=self.stopWords,ngram_range=(1,1))

            # Vectorize training emails
            transformedTrainingSet = countVectorizer.fit_transform(self.X_train)
            self.X_train=pd.DataFrame(transformedTrainingSet.toarray(),columns=countVectorizer.vocabulary_)

            # Vectorize test emails
            self.X_test = pd.DataFrame(countVectorizer.transform(self.X_test).toarray(),columns=countVectorizer.vocabulary_)
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            X_train,X_test,y_train,y_test=train_test_split(self.normalEmails+self.spamEmails,\
                                                                [1]*len(self.normalEmails)+[0]*len(self.spamEmails),\
                                                                test_size = 0.33,\
                                                                random_state = randomSeed)
            countVectorizer = CountVectorizer(tokenizer=jieba.lcut, stop_words=self.stopWords, ngram_range=(1, 1))
            # Vectorize training emails
            transformedTrainingSet = countVectorizer.fit_transform(X_train)
            X_train = pd.DataFrame(transformedTrainingSet.toarray(), columns=countVectorizer.vocabulary_)

            # Vectorize test emails
            X_test = pd.DataFrame(countVectorizer.transform(X_test).toarray(),
                                       columns=countVectorizer.vocabulary_)

            return X_train,X_test,y_train,y_test

    def estimateNaiveBayes(self,alpha,X_train,X_test,y_train,y_test,tagAsModelofClass=False):
        '''
        This method has no relationship with the class itself. No additional attributes are added to `self`
        '''
        if tagAsModelofClass==False: # don't train the Naive Bayes Model for the class (not inplace)
            naiveBayesModel=MultinomialNB(alpha=alpha)
            naiveBayesModel.fit(X_train,y_train)
            predictedLabels = naiveBayesModel.predict(X_test)
            show_metrics(true_labels=y_test,predicted_labels=predictedLabels,showMetrics=False)
            return naiveBayesModel,predictedLabels
        else:
            if X_train!=None or y_train!=None or X_test!=None or y_test!=None:
                msg = "class NaiveBayesSpamDetector method estimateNaiveBayes: X_train,X_test,y_train,y_test are not used, because`tagAsModelofClass` is set to True!"
                raise Exception(msg)
            self.naiveBayesModel = MultinomialNB(alpha=alpha)
            self.naiveBayesModel.fit(self.X_train, self.y_train)
            self.predictedLabels = self.naiveBayesModel.predict(self.X_test.values)
            show_metrics(true_labels=self.y_test, predicted_labels=self.predictedLabels,showMetrics=True)
            return self.naiveBayesModel, self.predictedLabels

    def displayROCofClass(self,showPlot:bool,outputPATH='./OutputResults'):
        fpr,tpr,thresholds = metrics.roc_curve(self.y_test,self.predictedLabels)
        roc_auc = metrics.auc(fpr,tpr)
        display = metrics.RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc,estimator_name=self.__class__.__name__)
        display.plot()
        if showPlot==True:plt.show()
        plt.savefig(os.path.join(outputPATH,'{}ROCcurve.jpg'.format(self.__class__.__name__)))

    def gridSearchParams(self,showPlot:bool,paramTup=[i/100 for i in range(0,110,10)],testTimes=10,outputPATH='./OutputResults'):
        '''grid search all available parameters, and save the result as figure.'''
        testJournal={
            'testId':[],
            'alpha':[],
            'roc_auc':[],
            'accuracy':[],
            'precision':[],
            'recall':[],
            'F1':[]
        }
        for testId in tqdm(range(testTimes)):
            X_train, X_test, y_train, y_test = self.createVector(randomSeed=int(time.time()),tagAsModelofClass=False)
            for alpha in paramTup:
                naiveBayesModel,predictedLabels=self.estimateNaiveBayes(alpha=alpha,X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, tagAsModelofClass=False)
                accuracy,precision,recall,F1 = show_metrics(true_labels=y_test, predicted_labels=predictedLabels, showMetrics=False)
                fpr, tpr, thresholds = metrics.roc_curve(y_test, predictedLabels)
                roc_auc = metrics.auc(fpr, tpr)

                testJournal['testId'].append(testId)
                testJournal['alpha'].append(alpha)
                testJournal['roc_auc'].append(roc_auc)
                testJournal['accuracy'].append(accuracy)
                testJournal['precision'].append(precision)
                testJournal['recall'].append(recall)
                testJournal['F1'].append(F1)
        self.gridSearchJournal = pd.DataFrame(testJournal).groupby('alpha').mean()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.heatmap(self.gridSearchJournal.iloc[:, 1:], cmap='Blues', annot=True, fmt='.4f',ax=ax)
        if showPlot==True: plt.show()
        plt.savefig(os.path.join(outputPATH,'{}GridSearchResults.jpg'.format(self.__class__.__name__)))
        return self.gridSearchJournal.iloc[:,1:]

    def trainFinalModel(self,alpha):
        '''Use the determined parameter to train the Naive Bayes Model for the class'''
        # privatize the transformer for raw text
        countVectorizer = CountVectorizer(tokenizer=jieba.lcut, stop_words=self.stopWords, ngram_range=(1, 1))
        self.countVectorizerModel = countVectorizer.fit(self.normalEmails + self.spamEmails)
        # joblib.dump(countVectorizerModel,'OutputResults/BOWCountVectorModel.m')
        # with open('OutputResults/BOWCountVectorModel.pickle','wb') as f:
        #     pickle.dump(countVectorizerModel,f)
        transformedX = self.countVectorizerModel.transform(self.normalEmails + self.spamEmails)
        y = [1]*len(self.normalEmails)+[0]*len(self.spamEmails)
        # privatize the Naive Bayes Model trained
        self.naiveBayesModel = MultinomialNB(alpha=alpha)
        self.naiveBayesModel.fit(transformedX,y)

        # joblib.dump(self.naiveBayesModel,'OutputResults/BOWSpamDetectorModel.m')
        # with open('OutputResults/BOWSpamDetectorModel.pickle', 'wb') as fw:
        #     pickle.dump(naiveBayesModel, fw)
        return self.naiveBayesModel

if __name__ == '__main__':
    myInstance = BOWSpamDetector(normalTextFilePATH='./Dataset/normal.txt',spamTextFilePATH='./Dataset/spam.txt', stopWordsTextFilePATH='./Dataset/stopwords_master/baidu_stopwords.txt')
    myInstance.preprocessEmails(ChineseOnly=False)
    gridSearchJournal = myInstance.gridSearchParams(showPlot=True,outputPATH='./OutputResults')