import numpy as np

class earlyStop:
    def __init__(self, numAllow, minDelta):
        self.patience = numAllow
        self.minDelta = minDelta
        self.stop = False
        self.minVal = np.inf
        self.count = 0

    def stopOrNot(self, valLoss):
        if (valLoss-self.minDelta) > self.minVal:
            self.count += 1
            if self.count > self.patience-1:
                self.stop = True
        elif valLoss < self.minVal:
            self.minVal = valLoss
            self.count = 0