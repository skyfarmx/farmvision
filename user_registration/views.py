
from django.shortcuts import render
from pathlib import Path
from django.contrib.auth import authenticate, login as auth_login, logout as log_out
from django.shortcuts import redirect
from yolowebapp2 import settings
from django.contrib.auth.forms import UserCreationForm #, AuthenticationForm, UsernameField
from .forms import UserForm, UsersForm
from dron_map.forms import Projects_Form
from user_registration.models import Users #, Projects
from django.shortcuts import get_object_or_404
BASE_DIR = Path(__file__).resolve().parent.parent

def user_pr(request, id):
    if request.user.is_authenticated:
        #
        url = get_object_or_404(Users, kat_id=id)
        return render(request, "user.html", {'userss': url, })
    else:
        return render(request, "login.html",)


def user_edit(request, id):
    if request.user.is_authenticated:
        model_user = Users.objects.get(kat_id=id)

        if request.method == 'POST':
            form = UsersForm(request.POST or None,
                             request.FILES or None, instance=model_user)

            if form.is_valid():
                form.picture = form.cleaned_data['picture']

                form.save()
            url = get_object_or_404(Users, kat_id=id)
            return render(request, "user_edit.html", {'userss': url, })
        else:
            url = get_object_or_404(Users, kat_id=id)
            return render(request, "user_edit.html", {'userss': url, })
    else:
        return render(request, "login.html",)
    



def login(request):
    if request.user.is_authenticated:
        return redirect('/')

    return render(request, "login.html",)


def login_view(request):
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        print(username, user, "00000000000000000000000000000000")
        if user is not None:
            if user.is_active:
                auth_login(request, user)
                return redirect('/', user)
        else:

            return redirect(settings.LOGOUT_REDIRECT_URL)

    else:
        return render(request, "login.html",)


def signup(request):

    if request.user.is_authenticated:
        return redirect('/')

    if request.method == 'POST':
        form = UserForm(request.POST,)
        print(form.errors, "bBBBBBBBBBBBBBBBBBBBB")
        if form.is_valid():
            form.save()
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']
            email = form.cleaned_data['email']
            user = authenticate(username=username,
                                password=password, email=email)
            auth_login(request, user)
            return redirect('/')

        else:
            return render(request, 'signup.html', {'form': form})

    else:
        form = UserCreationForm()
        return render(request, 'signup.html', {'form': form})

    #return render(request, "signup.html",)


def forgot_password(request):

    return render(request, "forgot_password.html",)


def logout(request):
    log_out(request)
    return redirect(settings.LOGOUT_REDIRECT_URL)


def calendar(request):
    if request.user.is_authenticated:
        userss = Users.objects.get(kat_id=request.user.id)
        return render(request, "calendar.html", {"userss": userss})
    else:
        return render(request, "login.html",)


def pricing(request):
    if request.user.is_authenticated:
        userss = Users.objects.get(kat_id=request.user.id)
        return render(request, "pricing.html", {"userss": userss})
    else:
        return render(request, "login.html",)