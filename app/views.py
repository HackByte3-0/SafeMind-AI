from django.shortcuts import render, redirect 
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from accounts.models import Profile
from app.models import TestResult,EmotionSessionData,ChatHistory
from collections import Counter
from app.forms import *
import cv2
from fer import FER
from django.http import StreamingHttpResponse
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
# Create your views here.


def index(request):
    return render(request, 'app/index.html')


def about(request):
    return render(request, 'app/about.html')

def contact(request):
    return render(request, 'app/contact.html')

@login_required
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





def submit_score(request):
    score = request.session.get('score', 0)
    emotions = request.session.get('emotions', [])  # Retrieve stored emotions
    if request.user.is_authenticated:
        TestResult.objects.create(user=request.user, phq9_score=score, emotions=emotions)
    request.session.flush()  # Reset the session after submitting
    return render(request, 'submit_score.html', {'score': score})


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.emotion_detector = FER()
        self.is_running = False

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        emotions = self.emotion_detector.detect_emotions(image)

        for emotion in emotions:
            (x, y, w, h) = emotion['box']
            dominant_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
            print(f"Detected Emotion: {dominant_emotion}")  # Log detected emotion for debugging

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def detect_emotions(self):
        if not self.is_running:
            return None
        success, image = self.video.read()
        emotions = self.emotion_detector.detect_emotions(image)
        if emotions:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            return dominant_emotion
        return None

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

def gen(camera):
    while camera.is_running:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    camera = VideoCamera()
    camera.start()
    return StreamingHttpResponse(gen(camera),
                                 content_type='multipart/x-mixed-replace; boundary=frame')




@login_required
def phq9_view(request):
    video_camera = VideoCamera()
    video_camera.start()

    # Retrieve or create an EmotionSessionData entry for the current user
    session_data, created = EmotionSessionData.objects.get_or_create(user=request.user)

    print(f"Initial emotion_score: {session_data.emotion_score}")
    print(f"Initial emotion_counts: {session_data.emotion_counts}")

    if request.method == 'POST':
        form = PHQ9Form(request.POST)
        if form.is_valid():
            form_score = sum(int(form.cleaned_data[q]) for q in form.cleaned_data)
            total_score = form_score + session_data.emotion_score

            # Determine depression status based on total score
            if total_score >= 20:
                result = "Severe Depression"
            elif total_score >= 15:
                result = "Moderately Severe Depression"
            elif total_score >= 10:
                result = "Moderate Depression"
            elif total_score >= 5:
                result = "Mild Depression"
            else:
                result = "Minimal or No Depression"

            video_camera.stop()

            # Get the dominant emotion
            dominant_emotion = max(session_data.emotion_counts, key=session_data.emotion_counts.get)
            recommendation = get_recommendation(form_score, result)
            # Save the result and dominant emotion
            TestResult.objects.create(
                user=request.user,
                phq9_score=form_score,
                total_score=total_score,
                Status=result,
                emotion_score=session_data.emotion_score,  # Save the overall emotion score
                emotions=session_data.emotion_counts  # Store all detected emotions and their counts
            )

            print(f"Final emotion_score: {session_data.emotion_score}")
            # Clear the session data from the database after saving
            session_data.delete()

            return render(request, 'app/result.html', {
                'score': total_score,
                'result': result,
                'emotion_score': session_data.emotion_score,
                'dominant_emotion': dominant_emotion,
                'recommendation': recommendation
            })
    else:
        form = PHQ9Form()

    # Detect emotions and update session data in the database
    dominant_emotion = video_camera.detect_emotions()
    if dominant_emotion in ['sad', 'angry', 'disgust', 'fear']:
        session_data.emotion_score += 1

    if dominant_emotion in session_data.emotion_counts:
        session_data.emotion_counts[dominant_emotion] += 1
    else:
        session_data.emotion_counts[dominant_emotion] = 1

    session_data.save()

    return render(request, 'app/phq9_form.html', {
        'form': form,
        'emotion_score': session_data.emotion_score
    })

@login_required
def chatbot_view(request):
    """Render the chat interface with history"""
    history = ChatHistory.objects.filter(user=request.user).order_by('-timestamp')[:10]
    
    # Add initial greeting if no history exists
    if not history.exists():
        initial_greeting = {
            'is_bot': True,
            'message': "🌼 Hi! I'm Mindbloom, your mental wellness companion. "
                      "I'm here to listen without judgment. How are you feeling today?",
        
        }
    else:
        initial_greeting = None
    
    return render(request, 'app/chatbot.html', {
        'history': history,
        'initial_greeting': initial_greeting
    })

@login_required
def chat(request):
    if request.method == 'POST':
        user_message = request.POST.get('message').strip()
        
        # Handle empty messages
        if not user_message:
            return JsonResponse({'response': "🌱 I'm here to listen. Please share what's on your mind."})

        # Enhanced prompt with conversation context
        prompt = f"""**You are Mindbloom** - a compassionate mental health companion. 
        **User says:** "{user_message}"

        **Response Rules:**
        1. Start with emotional validation
        2. Use plant/nature metaphors when possible 🌿
        3. Suggest one simple coping strategy
        4. Keep responses 2-3 sentences max
        5. Never diagnose - encourage professional help if needed
        6. Use warm, conversational tone with occasional emojis

        **Example Good Response:**
        "That sounds really tough, but I admire your strength in sharing this. 🌱 Sometimes our minds need stormy days to grow stronger. Would taking 3 deep breaths help right now?"

        **Now Craft Your Response:**"""
        
        try:
            # Generate response
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = model.generate_content(prompt)
            chat_response = response.text.strip().replace('**', '')  # Remove markdown
            
            # Save to history
            ChatHistory.objects.create(
                user=request.user,
                message=user_message,
                response=chat_response
            )
            
            return JsonResponse({'response': chat_response})
        
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return JsonResponse({
                'response': "🌧️ Hmm, my petals are feeling a bit droopy. Could you try rephrasing that?"
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def get_recommendation(score, category):
    prompt = f"Based on a PHQ-9 depression score of {score}, categorized as {category}, provide a brief 3-4 line recommendation for mental health care, focusing on self-care and professional advice."
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    if response and response.text:
        return response.text.strip().split("\n")[0:4]  # Limit to 3-4 lines
    return "No recommendation available."  # Return a default message if no response is generated