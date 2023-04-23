"""
    4/19/2023
    CS5001
    Zichong Meng
    This program file is for Final Project code to run fineTuning
"""

#import
from model import modelFinetuning

#main method for finetuning the model and test the fine tuned model
def main():
    f = modelFinetuning()
    f.tuning()
    f.testing()

if __name__ == '__main__':
    main()

