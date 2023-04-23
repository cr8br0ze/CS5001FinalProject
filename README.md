##### This is my final project for CS5001.
##### Installation requirement:
```bash
conda create -n CS5001FinalProject python=3.10
conda activate CS5001FinalProject
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

##### Download the dataset for this project
https://www.kaggle.com/datasets/kazanova/sentiment140
##### remember to donwload the dataset into the same folder as the code

##### To run finetuning
```bash
bash runFineTuning.sh
```
##### After finetuning
##### To run prediction
```bash
bash runPrediction.sh
```

##### Thank you
