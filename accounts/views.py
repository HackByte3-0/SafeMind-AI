
from django.shortcuts import redirect,render
# Create your views here.
from allauth.account.views import SignupView
from django.urls import reverse_lazy

class CustomSignupView(SignupView):
    def get_success_url(self):
        return reverse_lazy('account_login')  # Redirect to login after signup
    
    
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .forms import ProfileForm  # Create a form if needed
from accounts.models import Profile  # Import your Profile model


@login_required
def complete_profile_view(request):
    # Get or create the user's profile
    profile, created = Profile.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        form = ProfileForm(request.POST,request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            return redirect('dashboard')  # Redirect to dashboard/home after saving
    else:
        form = ProfileForm(instance=profile)  # Initialize form with profile data
    
    return render(request, 'accounts/complete_profile.html', {'form': form})