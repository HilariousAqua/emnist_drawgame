from kagglehub import dataset_download
from shutil import copy
import os

if not os.path.exists("data"):
    os.makedirs("data")

files = ["emnist-balanced-train.csv", "emnist-balanced-test.csv", "emnist-balanced-mapping.txt"]

for file in files:
    path = dataset_download('crawford/emnist', path=file)
    copy(path, f"data/{file}")
    
words_path = dataset_download('rtatman/english-word-frequency', path="unigram_freq.csv")
copy(words_path, "data/unigram_freq.csv")

