from django.shortcuts import render
from pathlib import Path
from django.core.files.storage import FileSystemStorage
import time
from farmvision import predict_tree, hashing
BASE_DIR = Path(__file__).resolve().parent.parent
from django.http import FileResponse
from dron_map.models import Users


def index(request):
    if request.user.is_authenticated:
        print(str(BASE_DIR), "BASE_DIRBASE_DIRBASE_DIRBASE_DIR")
        userss = Users.objects.get(kat_id=request.user.id)
        response = {}
        if request.method == 'POST':

            meyve_grubu = request.POST.get('meyve_grubu')
            agac_sayi = request.POST.get('agac_sayi')
            agac_yasi = request.POST.get('agac_yasi')
            ekilis_sira = request.POST.get('ekilis_sira')
            filename = request.FILES.get('file')
            
            if meyve_grubu is not None and agac_sayi is not None and agac_yasi is not None and ekilis_sira is not None and filename is not None:
                fs = FileSystemStorage(location=str(BASE_DIR)+"/static/images/")
                saved_filename = fs.save(filename.name, filename)
                start_time = time.time()
                
                if meyve_grubu == "mandalina":
                    if 0 < int(agac_yasi) <= 4:
                        detec = predict_tree.preddict(path_to_weights="mandalina.pt", path_to_source=f"{BASE_DIR}/static/images/{filename}")
                        response['kilo'] = int(detec[-3:-1].decode("utf-8")) * 0.125
                        response['count'] = detec[-3:-1].decode("utf-8")
                        response['toplam_ceki'] = int(agac_sayi) * response['kilo']
                        print(response['toplam_ceki'])

                    elif 4 < int(agac_yasi) <= 8:
                        pass
                    elif 8 < int(agac_yasi) <= 30:
                        pass

                elif meyve_grubu == "elma":
                    detec = predict_tree.preddict(path_to_weights="elma.pt", path_to_source=f"{BASE_DIR}/static/images/{filename}")
                    response['kilo'] = int(detec[-3:-1].decode("utf-8")) * 0.105
                    response['count'] = detec[-3:-1].decode("utf-8")
                    response['toplam_ceki'] = int(agac_sayi) * response['kilo']

                elif meyve_grubu == "armut":
                    detec = predict_tree.preddict(path_to_weights="armut.pt", path_to_source=f"{BASE_DIR}/static/images/{filename}")
                    response['kilo'] = int(detec[-3:-1].decode("utf-8")) * 0.220
                    response['count'] = detec[-3:-1].decode("utf-8")
                    response['toplam_ceki'] = int(agac_sayi) * response['kilo']

                elif meyve_grubu == "seftali":
                    detec = predict_tree.preddict(path_to_weights="seftali.pt", path_to_source=f"{BASE_DIR}/static/images/{filename}")
                    response['count'] = detec[-3:-1].decode("utf-8")
                    response['kilo'] = int(detec[-3:-1].decode("utf-8")) * 0.185
                    response['toplam_ceki'] = int(agac_sayi) * response['kilo']

                elif meyve_grubu == "nar":
                    detec = predict_tree.preddict(path_to_weights="nar.pt", path_to_source=f"{BASE_DIR}/static/images/{filename}")
                    response['count'] = detec[-3:-1].decode("utf-8")
                    response['kilo'] = int(detec[-3:-1].decode("utf-8")) * 0.300
                    response['toplam_ceki'] = int(agac_sayi) * response['kilo']

                elif meyve_grubu == "hurma":
                    detec = predict_tree.preddict(path_to_weights="hurma.pt", path_to_source=f"{BASE_DIR}/static/images/{filename}")
                    response['count'] = detec[-3:-1].decode("utf-8")

                response['time'] = f"{(time.time()-start_time):.2f}"
                response['image'] = f'images/{saved_filename}'
                response['image_detection'] = f'detected/{filename}'
                return render(request, "main.html", {"response": response, "userss": userss})
            else:
                return render(request, "main.html",)
        else:
            return render(request, "main.html", {"userss": userss})
    else:
        return render(request, "login.html",)


def multi_detection_image(request):
    if request.user.is_authenticated:
        print(str(BASE_DIR), "BASE_DIRBASE_DIRBASE_DIRBASE_DIR")
        userss = Users.objects.get(kat_id=request.user.id)
        response = {}
        if request.method == 'POST':

            meyve_grubu = request.POST.get('meyve_grubu')            
            ekilis_sira = request.POST.get('ekilis_sira')            
            filename = request.FILES.getlist('file')
            print(filename, "Filenamaaaaa")
            
            start_time = time.time()
            agac_sayi = 1
            
            if meyve_grubu == "mandalina":
                hass = hashing.add_prefix2(filename=f"{time.time()}") 
                fs = FileSystemStorage(location=str(hass[0])) 
                for image in filename:                    
                    saved_filename = fs.save(image.name, image)         
                    
                detec = predict_tree.multi_predictor(path_to_weights="mandalina.pt", path_to_source=hass[0], ekilis_sira=ekilis_sira, hashing=hass[1])

            elif meyve_grubu == "elma":
                hass = hashing.add_prefix2(filename=f"{time.time()}") 
                fs = FileSystemStorage(location=str(hass[0])) 
                for image in filename:                    
                    saved_filename = fs.save(image.name, image)
                    
                detec = predict_tree.multi_predictor(path_to_weights="elma.pt", path_to_source=hass[0], ekilis_sira=ekilis_sira, hashing=hass[1])

            elif meyve_grubu == "armut":
                hass = hashing.add_prefix2(filename=f"{time.time()}") 
                fs = FileSystemStorage(location=str(hass[0])) 
                for image in filename:                    
                    saved_filename = fs.save(image.name, image)
                    
                detec = predict_tree.multi_predictor(path_to_weights="armut.pt", path_to_source=hass[0], ekilis_sira=ekilis_sira, hashing=hass[1])

            elif meyve_grubu == "seftali":
                hass = hashing.add_prefix2(filename=f"{time.time()}") 
                fs = FileSystemStorage(location=str(hass[0])) 
                for image in filename:                    
                    saved_filename = fs.save(image.name, image)
                    
                detec = predict_tree.multi_predictor(path_to_weights="seftali.pt", path_to_source=hass[0], ekilis_sira=ekilis_sira, hashing=hass[1])

            elif meyve_grubu == "nar":
                hass = hashing.add_prefix2(filename=f"{time.time()}") 
                fs = FileSystemStorage(location=str(hass[0])) 
                for image in filename:                    
                    saved_filename = fs.save(image.name, image)
                    
                detec = predict_tree.multi_predictor(path_to_weights="nar.pt", path_to_source=hass[0], ekilis_sira=ekilis_sira, hashing=hass[1])

            elif meyve_grubu == "hurma":
                hass = hashing.add_prefix2(filename=f"{time.time()}") 
                fs = FileSystemStorage(location=str(hass[0])) 
                for image in filename:                    
                    saved_filename = fs.save(image.name, image)
                    
                detec = predict_tree.multi_predictor(path_to_weights="hurma.pt", path_to_source=hass[0], ekilis_sira=ekilis_sira, hashing=hass[1])

            return render(request, "multi_detection_fruit.html", {"response": hass[1], "userss": userss})
            
        else:
            return render(request, "multi_detection_fruit.html", {"userss": userss})
    else:
        return render(request, "login.html",)


def download_image(request, slug):
    print(slug, "sluggggggggggggggggg")
    return FileResponse(open(f"{BASE_DIR}/media/{slug}_result.zip", 'rb'), as_attachment=True)