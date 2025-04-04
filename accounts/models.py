from django.db import models
from django.contrib.auth.models import User
import uuid

class Profile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')  # Add this line
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/', null=True, blank=True)
    email = models.EmailField(max_length=100)
    date_of_birth = models.DateField(null=True, blank=True)  # Made this nullable for initial creation
    age = models.IntegerField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)  # Made nullable for initial creation
    Blood_Group = models.CharField(max_length=10, null=True, blank=True)
    weight = models.FloatField(null=True, blank=True)  # Made nullable for initial creation

    def __str__(self):  
        return self.first_name