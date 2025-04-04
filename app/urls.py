from django.urls import path
from app.views import *
from accounts.views import complete_profile_view

urlpatterns = [
    path('',index,name='home'),
    path('about/',about,name='about'),
    path('contact/',contact,name='contact'),
    path('dashboard/',dashboard,name='dashboard'),
    path('how-to-use/',how_to_use,name='how-to-use'),
    path('book-consultation/',book_consultation,name='book_consultation'),
    
]
