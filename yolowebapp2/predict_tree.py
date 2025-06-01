import subprocess,os,time
from pathlib import Path
import openpyxl
from natsort import natsorted 
import numpy as np
import glob
import zipfile
BASE_DIR = Path(__file__).resolve().parent.parent




def preddict(path_to_weights,path_to_source):
    os.chdir(f"{BASE_DIR}/detection/yolo")
    #os.setpgid()
    python_path = "/myprojects/myprojectenv/bin/python3"
    python_file = f"{BASE_DIR}/detection/yolo/detectcount.py" #"/home/murad/Belgeler/yolowebapp/detection/yolo/detectcount.py" 
    path_to_project= f"{BASE_DIR}/static"   #"/home/murad/Belgeler/yolowebapp/static"  
    detec = subprocess.check_output([python_path, python_file, "--weights", path_to_weights, "--conf", "0.1", "--img-size", "640", "--source", path_to_source, "--project", path_to_project, "--name", "detected"], timeout=600)
    #time.sleep(150)
    return detec

def multi_predictor(path_to_weights,path_to_source,ekilis_sira,hashing):
    print(ekilis_sira,type(ekilis_sira))
    a,b = ekilis_sira.split("-")
    print(path_to_source,"path_to_sourcepath_to_sourcepath_to_source")
    path_to_source_images = natsorted(glob.glob(f"{path_to_source}/*"))
    print(path_to_source_images,"path_to_source_imagespath_to_source_imagespath_to_source_images")

    exc_shit = list()
    os.chdir(f"{BASE_DIR}/detection/yolo")
    #os.setpgid()
    python_path = "/myprojects/myprojectenv/bin/python3"
    python_file = f"{BASE_DIR}/detection/yolo/detectcount.py" #"/home/murad/Belgeler/yolowebapp/detection/yolo/detectcount.py" 
    path_to_project= path_to_source   #"/home/murad/Belgeler/yolowebapp/static"  
    for images in path_to_source_images:
        print(images,"imageeeeeeeeeeessssssssss")
        detec = subprocess.check_output([python_path, python_file, "--weights", path_to_weights, "--conf", "0.1", "--img-size", "640", "--source", images, "--project", path_to_project, "--name", "detected"], timeout=6000)
        exc_shit.append(int(detec[-3:-1].decode("utf-8")))
    
    
    if not os.path.exists(f'{path_to_source}/excel'):
        os.makedirs(f'{path_to_source}/excel')
    data = np.array(exc_shit).reshape(int(a),int(b))
    wb = openpyxl.Workbook()
    ws = wb.active
    for i in data: 
        print(list(i))
        ws.append(list(i))
    wb.save(f'{path_to_source}/excel/output.xlsx')
    with zipfile.ZipFile(f"{BASE_DIR}/media/{hashing}_result.zip", mode="w") as archive:
        
        archive.write(f'{path_to_source}/excel/output.xlsx',"output.xlsx")
        for images in natsorted(glob.glob(f"{path_to_project}/detected/*")):
            print(images)
            archive.write(images,f"detected/{os.path.split(images)[-1]}")


    #return exc_shit

