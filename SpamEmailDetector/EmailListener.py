import datetime
import pickle
import time
import zmail
from SpamEmailDetector.TfIdfSpamDetector import TfIdfSpamDetector
from SpamEmailDetector.BOWSpamDetector import BOWSpamDetector
import argparse


class EmailListener:
    def __init__(self,email,password,detectorName):
        self.server = zmail.server(email,password) # loging in to mail server
        self.numberOfEmails = self.server.stat()[0] # record number of emails now
        # load specified detector
        if detectorName=='TfIdfSpamDetector':
            self.detector = TfIdfSpamDetector(normalTextFilePATH='./Dataset/normal.txt',
                                           spamTextFilePATH='./Dataset/spam.txt',
                                           stopWordsTextFilePATH='./Dataset/stopwords_master/baidu_stopwords.txt')
            self.detector.preprocessEmails(ChineseOnly=False)
            # gridSearchJournal = myInstance.gridSearchParams(showPlot=True,outputPATH='./OutputResults')
            self.detector.trainFinalModel(alpha=0.1, ngram_range=(1, 3))
        elif detectorName=='BOWSpamDetector':
            self.detector = BOWSpamDetector(normalTextFilePATH='./Dataset/normal.txt',spamTextFilePATH='./Dataset/spam.txt', stopWordsTextFilePATH='./Dataset/stopwords_master/baidu_stopwords.txt')
            self.detector.preprocessEmails(ChineseOnly=False)
            self.detector.trainFinalModel(alpha=0.1)
    def startListening(self,breakSeconds):
        print("Start Listening:")
        while True:
            print('-'*50,"\nListening at ",datetime.datetime.now(),end='\n'+'-'*50+'\n')
            if self.server.stat()[0]>self.numberOfEmails: # if number of emails in box increases
                print("A new e-mail received!")
                mail = self.server.get_latest()
                mailContent = mail['content_text']

                transformedX = self.detector.countVectorizerModel.transform(mailContent) # vectorize text
                prediction = self.detector.naiveBayesModel.predict(transformedX) # predict
                if prediction==1:
                    print('This is a normal email!')
                else:
                    print('This is a spam email!')
                self.numberOfEmails = self.server.stat()[0]
            time.sleep(breakSeconds)

if __name__ == '__main__':
    # listen to my email account
    # Note to use the POP3 or equivalent password instead of origin login password
    myEmail = EmailListener(email='xueyanhu1231@163.com',password='**********',detectorName='BOWSpamDetector')
    myEmail.startListening(10)