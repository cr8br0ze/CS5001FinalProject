"""
    4/19/2023
    CS5001
    Zichong Meng
    This program file is for Final Project model fineTuning
"""

#imports
from transformers import BertTokenizerFast, BertForSequenceClassification, get_scheduler
from textEmbedding import tokenize
from datasetPreprocess import twitterData
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from earlyStop import earlyStop

#the class for modelFinetuning
class modelFinetuning:

    #consturctor of the class
    #9 instance attributes
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 2).to(self.device)
        #self.model = torch.compile(self.model)
        #self.model = self.model.to(self.device)
        self.data = twitterData("training.1600000.processed.noemoticon.csv")
        self.max = self.data.trainMax
        self.epoch = 13
        self.stopper = earlyStop(2, 0)
        self.stopped = False
        self.tunedPath = list()

    #method take in text and its score to encode the text and then use them to create tensor dataset
    def tensorDataset(self, text, score):
        embedding = tokenize(text, self.max, self.tokenizer)
        input_ids, token_type_ids, attention_mask = embedding.encodeResult()
        score = torch.tensor(score.tolist()).to(self.device)
        return TensorDataset(input_ids, token_type_ids, attention_mask, score)

    #method takes tensor dataset and return a loaded pytorch dataloader set
    def loaddata(self, tensorDataset):
        return DataLoader(tensorDataset, batch_size=32)

    #the method for finetuning the model
    def tuning(self):
        trainTDataset  = self.tensorDataset(self.data.trainText, self.data.trainScore)
        trainLoaded = self.loaddata(trainTDataset)
        valTDataset = self.tensorDataset(self.data.valText, self.data.valScore)
        valLoaded = self.loaddata(valTDataset)

        totalSteps = len(trainLoaded) * self.epoch
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=totalSteps)

        for i in range(self.epoch):
            print(f"epoch{i}")

            #training
            print("training")
            self.model.train()
            self.trainStep(optimizer, lr_scheduler, trainLoaded)

            #validating
            print("validating")
            self.model.eval()
            valLoss = self.valStep(valLoaded)

            #saving model
            self.tokenizer.save_pretrained(f"epoch{i}")
            self.model.save_pretrained(f"epoch{i}")
            self.tunedPath.append(f"epoch{i}")

            #check whether to early stop to prevent overfitting
            self.stopper.stopOrNot(valLoss)
            if self.stopper.stopp():
                self.stopped = True
                print("early stopped")
                break

        print("tuning over")

    #the step to train the model in each epoch
    #takes in a optimizer, a scheduler and the training loaded dataLset
    def trainStep(self, optimizer, lr_scheduler, trainLoaded):
        totalTrainLoss = 0

        for step, batch in enumerate(tqdm(trainLoaded)):
            self.model.zero_grad()
            outputs = self.model(input_ids=batch[0].to(self.device),
                                 token_type_ids=batch[1].to(self.device),
                                 attention_mask=batch[2].to(self.device),
                                 labels=batch[3].to(self.device))
            loss = outputs.loss
            totalTrainLoss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

        print(f"training loss = {totalTrainLoss/len(trainLoaded)}")

    #the step to validate the model in each epoch after training
    #takes in the loaded validation dataset and return the validation loss
    def valStep(self, valLoaded):
        totalValLoss = 0
        probList = np.array([])

        for step, batch in enumerate(tqdm(valLoaded)):
            with torch.no_grad():
                outputs = self.model(input_ids=batch[0].to(self.device),
                                     token_type_ids=batch[1].to(self.device),
                                     attention_mask=batch[2].to(self.device),
                                     labels=batch[3].to(self.device))
            loss = outputs.loss
            totalValLoss += loss.item()
            predict = torch.argmax(outputs.logits, axis=1).flatten().to("cpu").numpy()
            score = batch[3].to("cpu").numpy()
            probability = (predict == score).mean()
            probList = np.append(probList, probability)

        print(f"validation loss = {totalValLoss / len(valLoaded)}")
        print(f"validation accuracy = {sum(probList) / len(valLoaded)}")
        return totalValLoss / len(valLoaded)

    #testing the model using the loaded testing set the best model
    def testing(self):
        #chosing the best model
        if self.stopped:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.tunedPath[len(self.tunedPath) - 3])
            self.model = BertForSequenceClassification.from_pretrained(self.tunedPath[len(self.tunedPath) - 3],
                                                                       num_labels=2).to(self.device)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.tunedPath[len(self.tunedPath)-1])
            self.model = BertForSequenceClassification.from_pretrained(self.tunedPath[len(self.tunedPath)-1],
                                                                       num_labels=2).to(self.device)

        testTDataset = self.tensorDataset(self.data.testText, self.data.testScore)
        testLoaded = self.loaddata(testTDataset)

        print("testing")
        self.model.eval()
        totalTestLoss = 0
        probList = np.array([])
        for step, batch in enumerate(tqdm(testLoaded)):
            with torch.no_grad():
                outputs = self.model(input_ids=batch[0].to(self.device),
                                     token_type_ids=batch[1].to(self.device),
                                     attention_mask=batch[2].to(self.device),
                                     labels=batch[3].to(self.device))
            loss = outputs.loss
            totalTestLoss += loss.item()
            predict = torch.argmax(outputs.logits, dim=1).flatten().to("cpu").numpy()
            score = batch[3].to("cpu").numpy()
            probability = (predict == score).mean()
            probList = np.append(probList, probability)

        print(f"testing loss = {totalTestLoss / len(testLoaded)}")
        print(f"testing accuracy = {sum(probList) / len(testLoaded)}")
