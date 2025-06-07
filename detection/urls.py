from django.urls import path, re_path
from . import views

app_name = 'detection'

urlpatterns = [
    # Ana sayfa - tekli meyve tespiti
    path('', views.index, name='index'),
    path('index/', views.index, name='index_alt'),
    
    # Çoklu görüntü analizi  
    path('multi-detection/', views.multi_detection_image, name='multi_detection_image'),
    
    # Dosya indirme
    re_path(r'^download_image/(?P<slug>[\w-]+)/$', views.download_image, name='download_image'),
]