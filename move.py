import shutil
import os

dir = "MoNuSegTestData/"
files = os.listdir(dir)

print(files)

if not os.path.exists("MoNuSegTestData/Tissue Images"):
    os.mkdir("MoNuSegTestData/Tissue Images")
    
if not os.path.exists("MoNuSegTestData/Annotations"):
    os.mkdir("MoNuSegTestData/Annotations")

for file in files:
    if file.endswith(".tif"):
        shutil.move("MoNuSegTestData/" + file, "MoNuSegTestData/Tissue Images/" + file)
    elif file.endswith(".xml"):
        shutil.move("MoNuSegTestData/" + file, "MoNuSegTestData/Annotations/" + file)
