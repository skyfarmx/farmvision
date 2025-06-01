from django.urls import path, re_path

from . import views
app_name = 'user_registration'



urlpatterns = [
   
    re_path(r'user_pr/(?:id-(?P<id>[0-9]+)/)?$', views.user_pr, name='user_pr'),          
    re_path(r'user_edit/(?:id-(?P<id>[0-9]+)/)?$', views.user_edit, name='user_edit'),          
    path('logout', views.logout, name='logout'),
    path('pricing', views.pricing, name='pricing'),
    path('calendar', views.calendar, name='calendar'),
    path('login', views.login, name='login'),      
    path('login_view', views.login_view, name='login_view'),      
    path('signup', views.signup, name='signup'),    
    path('forgot_password', views.forgot_password, name='forgot_password'), 

]

