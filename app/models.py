from django.db import models
import uuid
from django.contrib.auth.models import User
# Create your models here.



# models.py
from django.db import models
from django.contrib.auth.models import User

class TestResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    phq9_score = models.IntegerField()
    total_score = models.IntegerField(null=True, blank=True)
    Status = models.CharField(max_length=100, null=True, blank=True)
    emotions = models.JSONField(null=True,blank=True)  # Store the detected emotions
    date= models.DateTimeField(auto_now_add=True,null=True,blank=True)
    emotion_score = models.IntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.phq9_score}"
    
    class Meta:
        ordering = ['-date']


# Temproary model for storing session data

class EmotionSessionData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    emotion_score = models.IntegerField(default=0)
    emotion_counts = models.JSONField(default=dict)  # Storing emotion counts as JSON
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Emotion data for {self.user}"





class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)