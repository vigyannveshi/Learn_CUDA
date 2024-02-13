import os

files=os.listdir()


for file in files:
    if "." not in file:
        os.remove(file)