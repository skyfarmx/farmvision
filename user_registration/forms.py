from django import forms
from django.contrib.auth.forms import UserCreationForm  
from user_registration.models import Users
from django.contrib import admin



class UserForm(UserCreationForm):
    email = forms.EmailField(max_length=200) 
    USERNAME_FIELD = 'email' 


class UsersForm(forms.ModelForm):
	class Meta:
		model = Users
		fields = '__all__'


admin.site.register(Users)







