import shutil
import os

dir = "./data/MoNuSegTestData/"
files = os.listdir(dir)

print(files)

if not os.path.exists("./data/MoNuSegTestData/Tissue Images"):
    os.mkdir("./data/MoNuSegTestData/Tissue Images")
    
if not os.path.exists("./data/MoNuSegTestData/Annotations"):
    os.mkdir("./data/MoNuSegTestData/Annotations")

for file in files:
    if file.endswith(".tif"):
        shutil.move("./data/MoNuSegTestData/" + file, "./data/MoNuSegTestData/Tissue Images/" + file)
    elif file.endswith(".xml"):
        shutil.move("./data/MoNuSegTestData/" + file, "./data/MoNuSegTestData/Annotations/" + file)
