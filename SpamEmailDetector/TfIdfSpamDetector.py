import os
import pickle
import time
import seaborn as sns
import jieba
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from sklearn import metrics
from SpamEmailDetector.BOWSpamDetector import BOWSpamDetector
from SpamEmailDetector.Globals.GlobalFunctions import show_metrics
import joblib


class TfIdfSpamDetector(BOWSpamDetector):
    def createVector(self,ngram_range,randomSeed=2020310221,tagAsModelofClass=False):
        if tagAsModelofClass==True:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.normalEmails+self.spamEmails,\
                                                                [1]*len(self.normalEmails)+[0]*len(self.spamEmails),\
                                                                test_size = 0.33,\
                                                                random_state = randomSeed)
            # use TFIDF to train the data
            vectorizer = TfidfVectorizer(min_df=1,use_idf=True,ngram_range=ngram_range,tokenizer=jieba.lcut)
            # Vectorize training emails
            transformedTrainingSet = vectorizer.fit_transform(self.X_train)
            self.X_train = pd.DataFrame(transformedTrainingSet.toarray(), columns=vectorizer.vocabulary_)

            # Vectorize test emails
            self.X_test = pd.DataFrame(vectorizer.transform(self.X_test).toarray(),
                                       columns=vectorizer.vocabulary_)
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            X_train,X_test,y_train,y_test=train_test_split(self.normalEmails+self.spamEmails,\
                                                                [1]*len(self.normalEmails)+[0]*len(self.spamEmails),\
                                                                test_size = 0.33,\
                                                                random_state = randomSeed)
            vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, stop_words=self.stopWords, ngram_range=ngram_range)
            # Vectorize training emails
            transformedTrainingSet = vectorizer.fit_transform(X_train)
            X_train = pd.DataFrame(transformedTrainingSet.toarray(), columns=vectorizer.vocabulary_)

            # Vectorize test emails
            X_test = pd.DataFrame(vectorizer.transform(X_test).toarray(),
                                       columns=vectorizer.vocabulary_)
            return X_train,X_test,y_train,y_test

    def estimateNaiveBayes(self,alpha,X_train,X_test,y_train,y_test,tagAsModelofClass=False):
        '''
        This method has no relationship with the class itself. No additional attributes are added to `self`
        '''
        if tagAsModelofClass==False: # training result will not be tagged as belonging to class
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


    def gridSearchParams(self,showPlot:bool,paramTup_alpha=[i/100 for i in range(0,110,10)],paramTup_ngramRange=[(1,1),(1,2),(2,2),(1,3),(2,3),(3,3)],testTimes=10,outputPATH='./OutputResults'):
        testJournal={
            'ngram_range':[],
            'testId':[],
            'alpha':[],
            'roc_auc':[],
            'accuracy':[],
            'precision':[],
            'recall':[],
            'F1':[]
        }
        for ngram_range in tqdm(paramTup_ngramRange):
            for testId in range(testTimes):
                X_train, X_test, y_train, y_test = self.createVector(ngram_range=ngram_range,randomSeed=int(time.time()),tagAsModelofClass=False)
                for alpha in paramTup_alpha:
                    naiveBayesModel,predictedLabels=self.estimateNaiveBayes(alpha=alpha,X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, tagAsModelofClass=False)
                    accuracy,precision,recall,F1 = show_metrics(true_labels=y_test, predicted_labels=predictedLabels, showMetrics=False)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictedLabels)
                    roc_auc = metrics.auc(fpr, tpr)

                    testJournal['ngram_range'].append(ngram_range)
                    testJournal['testId'].append(testId)
                    testJournal['alpha'].append(alpha)
                    testJournal['roc_auc'].append(roc_auc)
                    testJournal['accuracy'].append(accuracy)
                    testJournal['precision'].append(precision)
                    testJournal['recall'].append(recall)
                    testJournal['F1'].append(F1)
        self.gridSearchJournal = pd.DataFrame(testJournal).groupby(['alpha','ngram_range']).mean()

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        sns.heatmap(myInstance.gridSearchJournal.iloc[:, 1:], cmap='Blues', annot=True, fmt='.4f', ax=ax)
        if showPlot==True: plt.show()
        plt.savefig(os.path.join(outputPATH,'{}GridSearchResults.jpg'.format(self.__class__.__name__)))
        return self.gridSearchJournal.iloc[:,1:]

    def trainFinalModel(self,alpha,ngram_range):
        self.countVectorizer = TfidfVectorizer(tokenizer=jieba.lcut, stop_words=self.stopWords, ngram_range=ngram_range)
        self.countVectorizerModel = self.countVectorizer.fit(self.normalEmails+self.spamEmails)
        transformedX = self.countVectorizerModel.transform(self.normalEmails+self.spamEmails)
        y = [1]*len(self.normalEmails)+[0]*len(self.spamEmails)
        self.naiveBayesModel = MultinomialNB(alpha=alpha)
        self.naiveBayesModel.fit(transformedX,y)
        # with open('OutputResults/TfidfSpamDetectorModel.pickle', 'wb') as fw:
        #     pickle.dump(self.naiveBayesModel, fw)
        return self.naiveBayesModel

if __name__ == '__main__':
    myInstance=TfIdfSpamDetector(normalTextFilePATH='./Dataset/normal.txt',spamTextFilePATH='./Dataset/spam.txt', stopWordsTextFilePATH='./Dataset/stopwords_master/baidu_stopwords.txt')
    myInstance.preprocessEmails(ChineseOnly=False)
    # gridSearchJournal = myInstance.gridSearchParams(showPlot=True,outputPATH='./OutputResults')
    myInstance.trainFinalModel(alpha=0.1,ngram_range=(1,3))
    # joblib.dump(myInstance, 'OutputResults/TfIdfSpamDetector')
    # with open('OutputResults/TfIdfSpamDetector.pickle','wb') as f:
    #     pickle.dump(myInstance,f)