"""
    4/21/2023
    CS5001
    Zichong Meng
    This program file is for Final Project a checker
    if the model need to early stop to prevent overfitting
"""

#imports
import numpy as np

#the earlyStop class
class earlyStop:

    #consturctor to create a early stopper
    #takes in number of consecutive epochs having increasing valLoss allowed
    #and the minDelta used to calculate if the valLoss is increased to a certain point or not
    def __init__(self, numAllow, minDelta):
        self.patience = numAllow
        self.minDelta = minDelta
        self.stop = False
        self.minVal = np.inf
        self.count = 0

    #takes in the current valLoss and return whether it should stop early or not
    def stopOrNot(self, valLoss):
        if (valLoss-self.minDelta) > self.minVal:
            self.count += 1
            if self.count > self.patience-1:
                self.stop = True
        elif valLoss < self.minVal:
            self.minVal = valLoss
            self.count = 0
        else:
            self.count = 0

    #return the decision to stop or not
    def stopp(self):
        return self.stop
