"""
    4/19/2023
    CS5001
    Zichong Meng
    This program file is for Final Project sentiment prediction
"""

#imports
from transformers import pipeline
import torch
import tkinter as tk

#the prediction class
class prediction():

    #constructor to create a predictor using the best model
    #when I finetuned the model, the epoch 3 model is the best
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(task="sentiment-analysis", model="epoch3", tokenizer="epoch3", device=device)
        self.result = list()

    #method for prediction
    #takes in aninput and return if it is positive or negative
    def predict(self, input):
        result = "positive" if self.pipe(input)[0]["label"] == 'LABEL_1' else "negative"
        self.result.append({input:result})
        return result

    #show the history of the prediction
    def history(self):
        return self.result

    #print the object which will print the history of the prediction
    def __str__(self):
        word = "your history is:\n"
        for i in self.result:
            for a,b in i.items():
                word = word + "input: "+a+" , result: "+b+"\n"
        return word

#main function for user interface for prediction
def main():
    predict = prediction()

    window = tk.Tk()
    window.title("Sentiment Predictor")
    window.geometry("550x325")

    label = tk.Label(window, text="Sentiment Predictor", font=("Times New Roman", 12), width=30, height=3)
    label.pack()
    box = tk.Entry(window, width=50, font=("Times New Roman", 12))
    box.pack()

    result = tk.Label(window, text="")
    result.pack()
    def pred():
        input = box.get()
        out = predict.predict(input)
        result.config(text=f"it is a {out} sentence")
    button = tk.Button(window, text='run', command=pred)
    button.pack()

    def newWindow():
        window2 = tk.Toplevel(window)
        window2.title("Sentiment Prediction History")
        window2.geometry("550x325")
        for i in predict.history():
            tk.Label(window2, text=i).pack()
    button2 = tk.Button(window, text='see history', command=newWindow)
    button2.pack()


    window.mainloop()

    print("prediction ended")
    print(predict)

if __name__ == '__main__':
    main()
