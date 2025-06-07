from django.shortcuts import render
from pathlib import Path
from django.core.files.storage import FileSystemStorage
import time
from farmvision import predict_tree, hashing
from django.http import FileResponse, Http404
from dron_map.models import Users
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect
from django.conf import settings
import os
import logging

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent

def sanitize_filename(filename):
    # Only keep safe characters in filename
    import re
    filename = os.path.basename(filename)
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

def get_count_from_prediction(detec):
    # Use a safer way to extract the count from the model output
    # Assumes detec is a dict or structured output, adjust as needed
    if hasattr(detec, 'get'):  # dict-like
        return int(detec.get('count', 0))
    elif isinstance(detec, (bytes, bytearray)):
        # fallback for legacy code, but should be replaced
        try:
            return int(detec[-3:-1].decode('utf-8'))
        except Exception:
            return 0
    elif isinstance(detec, str):
        # parse last numbers in string
        import re
        match = re.search(r'(\d+)', detec)
        return int(match.group(1)) if match else 0
    else:
        return 0

@csrf_protect
@login_required
def index(request):
    try:
        userss = Users.objects.get(kat_id=request.user.id)
    except Users.DoesNotExist:
        logger.error(f"User {request.user.id} not found in Users table.")
        return render(request, "login.html")

    response = {}
    if request.method == 'POST':
        meyve_grubu = request.POST.get('meyve_grubu')
        agac_sayi = request.POST.get('agac_sayi')
        agac_yasi = request.POST.get('agac_yasi')
        ekilis_sira = request.POST.get('ekilis_sira')
        file = request.FILES.get('file')
        
        if not all([meyve_grubu, agac_sayi, agac_yasi, ekilis_sira, file]):
            return render(request, "main.html", {"userss": userss, "error": "All fields are required."})

        if file.size > 20*1024*1024:
            return render(request, "main.html", {"userss": userss, "error": "File is too large."})

        safe_filename = sanitize_filename(file.name)
        images_dir = BASE_DIR / "static" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        fs = FileSystemStorage(location=str(images_dir))

        saved_filename = fs.save(safe_filename, file)
        full_image_path = images_dir / saved_filename

        start_time = time.time()
        try:
            if meyve_grubu == "mandalina":
                if 0 < int(agac_yasi) <= 4:
                    detec = predict_tree.preddict(path_to_weights="mandalina.pt", path_to_source=str(full_image_path))
                    count = get_count_from_prediction(detec)
                    response['kilo'] = count * 0.125
                    response['count'] = count
                    response['toplam_ceki'] = int(agac_sayi) * response['kilo']
                elif 4 < int(agac_yasi) <= 8:
                    return render(request, "main.html", {"userss": userss, "error": "Detection for this age range not implemented."})
                elif 8 < int(agac_yasi) <= 30:
                    return render(request, "main.html", {"userss": userss, "error": "Detection for this age range not implemented."})
            elif meyve_grubu == "elma":
                detec = predict_tree.preddict(path_to_weights="elma.pt", path_to_source=str(full_image_path))
                count = get_count_from_prediction(detec)
                response['kilo'] = count * 0.105
                response['count'] = count
                response['toplam_ceki'] = int(agac_sayi) * response['kilo']
            elif meyve_grubu == "armut":
                detec = predict_tree.preddict(path_to_weights="armut.pt", path_to_source=str(full_image_path))
                count = get_count_from_prediction(detec)
                response['kilo'] = count * 0.220
                response['count'] = count
                response['toplam_ceki'] = int(agac_sayi) * response['kilo']
            elif meyve_grubu == "seftali":
                detec = predict_tree.preddict(path_to_weights="seftali.pt", path_to_source=str(full_image_path))
                count = get_count_from_prediction(detec)
                response['count'] = count
                response['kilo'] = count * 0.185
                response['toplam_ceki'] = int(agac_sayi) * response['kilo']
            elif meyve_grubu == "nar":
                detec = predict_tree.preddict(path_to_weights="nar.pt", path_to_source=str(full_image_path))
                count = get_count_from_prediction(detec)
                response['count'] = count
                response['kilo'] = count * 0.300
                response['toplam_ceki'] = int(agac_sayi) * response['kilo']
            elif meyve_grubu == "hurma":
                detec = predict_tree.preddict(path_to_weights="hurma.pt", path_to_source=str(full_image_path))
                count = get_count_from_prediction(detec)
                response['count'] = count
            else:
                return render(request, "main.html", {"userss": userss, "error": "Unknown fruit group."})
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return render(request, "main.html", {"userss": userss, "error": "Prediction failed."})

        response['time'] = f"{(time.time()-start_time):.2f}"
        response['image'] = f'images/{saved_filename}'
        response['image_detection'] = f'detected/{saved_filename}'
        return render(request, "main.html", {"response": response, "userss": userss})
    else:
        return render(request, "main.html", {"userss": userss})

@csrf_protect
@login_required
def multi_detection_image(request):
    try:
        userss = Users.objects.get(kat_id=request.user.id)
    except Users.DoesNotExist:
        logger.error(f"User {request.user.id} not found in Users table.")
        return render(request, "login.html")

    if request.method == 'POST':
        meyve_grubu = request.POST.get('meyve_grubu')
        ekilis_sira = request.POST.get('ekilis_sira')
        file_list = request.FILES.getlist('file')

        if not (meyve_grubu and ekilis_sira and file_list):
            return render(request, "multi_detection_fruit.html", {"userss": userss, "error": "All fields are required."})

        for f in file_list:
            if f.size > 20*1024*1024:
                return render(request, "multi_detection_fruit.html", {"userss": userss, "error": f"File {f.name} is too large."})

        start_time = time.time()
        agac_sayi = 1

        try:
            hass = hashing.add_prefix2(filename=f"{time.time()}")
            fruit_dir = Path(hass[0])
            fruit_dir.mkdir(parents=True, exist_ok=True)
            fs = FileSystemStorage(location=str(fruit_dir))
            for image in file_list:
                safe_name = sanitize_filename(image.name)
                fs.save(safe_name, image)
            
            weights_map = {
                "mandalina": "mandalina.pt",
                "elma": "elma.pt",
                "armut": "armut.pt",
                "seftali": "seftali.pt",
                "nar": "nar.pt",
                "hurma": "hurma.pt"
            }
            if meyve_grubu not in weights_map:
                return render(request, "multi_detection_fruit.html", {"userss": userss, "error": "Unknown fruit group."})

            detec = predict_tree.multi_predictor(
                path_to_weights=weights_map[meyve_grubu],
                path_to_source=str(fruit_dir),
                ekilis_sira=ekilis_sira,
                hashing=hass[1]
            )
        except Exception as e:
            logger.error(f"Multi-prediction failed: {e}")
            return render(request, "multi_detection_fruit.html", {"userss": userss, "error": "Prediction failed."})

        return render(request, "multi_detection_fruit.html", {"response": hass[1], "userss": userss})
    else:
        return render(request, "multi_detection_fruit.html", {"userss": userss})

@login_required
def download_image(request, slug):
    file_path = BASE_DIR / "media" / f"{slug}_result.zip"
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise Http404("File not found.")
    try:
        return FileResponse(open(file_path, 'rb'), as_attachment=True)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise Http404("File could not be downloaded.")
