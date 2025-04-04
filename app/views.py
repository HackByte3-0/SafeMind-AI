from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from accounts.models import Profile
from app.models import TestResult

# Create your views here.


def index(request):
    return render(request, 'app/index.html')


def about(request):
    return render(request, 'app/about.html')

def contact(request):
    return render(request, 'app/contact.html')

def dashboard(request):
    profile = Profile.objects.get(email=request.user.email)
    results = TestResult.objects.filter(user=request.user).order_by('-date')

    context = {
        'profile': profile,
        'results': results,
    }

    return render(request, "app/dashboard.html",context)

def how_to_use(request):
    return render(request, 'app/how_to_use.html')

def book_consultation(request):
    return render(request, 'app/book_consultation.html')




