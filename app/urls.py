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
    path('video_feed/', video_feed, name='video_feed'),
    path("assesment/", phq9_view, name="phq9"),
    path('chatbot/', chatbot_view, name='chatbot'),
    path('chat/', chat, name='chat_api'),
    path('audio-phase/', audio_phase, name='audio_phase'),
    path('analyze-audio/', analyze_audio, name='analyze_audio'),
    path('results/<int:result_id>/', final_results, name='final_results'),
    path('journal/', journal, name='journal'),
    
]
