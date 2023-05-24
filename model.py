from trdg.generators import GeneratorFromStrings
import cv2
import numpy as np
import csv
from PIL import Image

DIV = "÷"
MUL = "×"
ADD = "+"
SUB = "—"

def generate_dataset():
    m1 = np.random.randint(1000, 10000, size=50)
    m2 = np.random.randint(100, 1000, size=25)
    m3 = np.random.randint(10, 100, size=15)
    m4 = np.array(list(range(1, 10)))

    mul = np.concatenate((m1, m2, m3, m4))

    mul = list(map(lambda n: str(n) + MUL, mul))
    div = list(map(lambda n: str(n) + DIV, list(range(1, 10)))) * 10
    add = list(map(lambda n: str(n) + ADD, list(range(1, 50)))) * 2
    sub = list(map(lambda n: str(n) + SUB, list(range(1, 9))))  * 10

    generator = GeneratorFromStrings(
        mul+div+add+sub
    )
    
    with open("dataset/labels.csv", "w") as f:

        writer = csv.writer(f)
        writer.writerow(["filename", "words"])

        for index, (img, lbl) in enumerate(generator):

            filename = f"dataset/{index}.png"
            img.save(filename)
            writer.writerow([filename, lbl])

if __name__ == "__main__":
    generate_dataset()