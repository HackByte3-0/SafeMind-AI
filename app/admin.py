from django.contrib import admin
from app.models import TestResult, EmotionSessionData ,ChatHistory,JournalEntry
# Register your models here.

admin.site.register(TestResult)
admin.site.register(EmotionSessionData)
admin.site.register(ChatHistory)
admin.site.register(JournalEntry)