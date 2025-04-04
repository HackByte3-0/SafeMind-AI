from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required


# Create your views here.


def index(request):
    return render(request, 'app/index.html')


def about(request):
    return render(request, 'app/about.html')

def contact(request):
    return render(request, 'app/contact.html')

def dashboard(request):
    return render(request, 'app/dashboard.html')

def how_to_use(request):
    return render(request, 'app/how_to_use.html')

def book_consultation(request):
    return render(request, 'app/book_consultation.html')




