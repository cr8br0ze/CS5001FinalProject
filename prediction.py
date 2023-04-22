from transformers import pipeline
import torch
import tkinter as tk


class prediction():
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(task="sentiment-analysis", model="epoch3", tokenizer="epoch3", device=device)
        self.result = list()

    def predict(self, input):
        result = "postive" if self.pipe(input)[0]["label"] == 'LABEL_1' else "negative"
        self.result.append({input:result})
        return result

    def history(self):
        return self.result

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

if __name__ == '__main__':
    main()
