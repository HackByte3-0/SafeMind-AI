import json
import traceback
from django.shortcuts import render, redirect ,get_object_or_404
from django.http import Http404, JsonResponse
from django.utils.safestring import mark_safe
from django.contrib.auth.decorators import login_required
from accounts.models import Profile
from app.models import TestResult,EmotionSessionData,ChatHistory
from collections import Counter
from app.forms import *
import cv2
from fer import FER
from django.http import StreamingHttpResponse
from django.urls import reverse
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import requests
from django.conf import settings
from django.contrib import messages

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
    profile = get_object_or_404(Profile,user=request.user)
    results = TestResult.objects.filter(user=request.user).order_by('-date')
    
    entries = JournalEntry.objects.filter(user=request.user).order_by('entry_date')
    
    chart_data = {
        'dates': [e.entry_date.strftime("%Y-%m-%d") for e in entries],
        'positive': [e.positive_score for e in entries],
        'negative': [e.negative_score for e in entries]
    }
    
    chart_data_json = mark_safe(json.dumps(chart_data))
    

    height_m = profile.height / 100

    # Calculate BMI
    bmi = profile.weight / (height_m ** 2)
    context = {
        'profile': profile,
        'results': results,
        'bmi': bmi,
        'chart_data_json': chart_data_json
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
    # Initialize video camera and emotion detection
    video_camera = VideoCamera()
    video_camera.start()
    logger.debug("PHQ-9 view accessed")
    # Get or create temporary emotion session
    session_data, created = EmotionSessionData.objects.get_or_create(
        user=request.user,
        defaults={'emotion_counts': {}, 'emotion_score': 0}
    )

    if request.method == 'POST':
        form = PHQ9Form(request.POST)
        if form.is_valid():
            # Calculate PHQ-9 score
            form_score = sum(int(form.cleaned_data[q]) for q in form.cleaned_data)
            
            # Calculate emotion contribution (your existing logic)
            emotion_contribution = session_data.emotion_score
            total_score = form_score + emotion_contribution
            
            # Get depression status
            depression_status = get_depression_status(total_score)
            
            # Store all data in session for audio phase
            request.session['phq9_data'] = {  # Changed key to match audio_phase check
                'form_score': form_score,
                'total_score': total_score,
                'depression_status': depression_status,
                'emotion_counts': session_data.emotion_counts,
                'emotion_score': session_data.emotion_score,
                'dominant_emotion': max(
                    session_data.emotion_counts.items(), 
                    key=lambda x: x[1]
                )[0] if session_data.emotion_counts else 'neutral'}
            
            # Cleanup resources
            video_camera.stop()
            session_data.delete()
            
            # Redirect to audio phase
            logger.debug("Form valid, redirecting to audio phase")
            return redirect('audio_phase')

        else:
            # Handle invalid form
            video_camera.stop()
            return render(request, 'app/phq9_form.html', {'form': form})
    
    else:
        form = PHQ9Form()
    
    # Render PHQ-9 test page with live emotion detection
    return render(request, 'app/phq9_form.html', {
        'form': form,
        'video_feed_url': reverse('video_feed')
    })


def get_depression_status(score):
    if score >= 20:
        return "Severe Depression"
    elif score >= 15:
        return "Moderately Severe Depression"
    elif score >= 10:
        return "Moderate Depression"
    elif score >= 5:
        return "Mild Depression"
    return "Minimal or No Depression"



def get_current_emotion(request):
    if request.user.is_authenticated:
        try:
            session_data = EmotionSessionData.objects.get(user=request.user)
            return JsonResponse({
                'emotion': max(
                    session_data.emotion_counts.items(),
                    key=lambda x: x[1]
                )[0] if session_data.emotion_counts else 'neutral'
            })
        except EmotionSessionData.DoesNotExist:
            return JsonResponse({'emotion': 'neutral'})
    return JsonResponse({'emotion': 'neutral'})

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





def audio_phase(request):
    # Verify session data exists
    if not request.session.get('phq9_data'):
        return redirect('phq9')
        
    return render(request, 'app/audio_recording.html')



def analyze_text_with_model(text):
    """Process Cloudflare API response and calculate depression score"""
    logger = logging.getLogger(__name__)
    logger.info("Starting text analysis")
    
    MODEL = "@cf/huggingface/distilbert-sst-2-int8"
    API_KEY = os.getenv("CLOUDFLARE_API_TOKEN") 
    ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    
    logger.info(f"API Config - Model: {MODEL}, Account ID exists: {bool(ACCOUNT_ID)}, API Key exists: {bool(API_KEY)}")
    
    try:
        API_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        logger.info(f"Making API request to {API_BASE_URL}{MODEL}")
        
        response = requests.post(
            f"{API_BASE_URL}{MODEL}",
            headers=headers,
            json={"text": text},
            timeout=10
        )
        
        logger.info(f"API Response status: {response.status_code}")
        
        # Check for API errors
        if response.status_code != 200:
            logger.error(f"API Error: HTTP {response.status_code} - {response.text}")
            raise ValueError(f"API Error: HTTP {response.status_code}")
            
        response_json = response.json()
        logger.info(f"API Response: {json.dumps(response_json)}")
        
        # Validate successful response
        if not response_json.get("success", False):
            logger.error(f"API request unsuccessful: {json.dumps(response_json)}")
            raise ValueError("API request unsuccessful")
        
        # Based on the exact response format you provided
        results = response_json.get("result", [])
        logger.info(f"Extracted results: {results}")
        
        # Extract negative sentiment score directly from the results list
        negative_score = 0.0
        for item in results:
            if item.get("label") == "NEGATIVE":
                negative_score = item.get("score", 0.0)
                logger.info(f"Found negative score: {negative_score}")
                break
        
        logger.info(f"Final negative score: {negative_score}")

        # Calculate base depression score (0-25 scale)
        adjusted_score = negative_score ** 0.7  # Non-linear scaling
        depression_score = round(adjusted_score * 25)
        logger.info(f"Base depression score: {depression_score}")
        
        # Apply keyword boosts (0-10 max boost)
        depression_keywords = ["sad", "depressed", "hopeless", "worthless",
                             "suffer", "can't feel", "pain", "tired",
                             "exhausted", "give up"]
        keyword_matches = sum(1 for kw in depression_keywords if kw in text.lower())
        logger.info(f"Keyword matches: {keyword_matches}")
        
        depression_score = min(depression_score + (keyword_matches * 2), 25)
        logger.info(f"Final depression score after keyword boost: {depression_score}")

        result = {
            'depression_score': depression_score,
            'confidence': negative_score,
            'processed_text': text,
            'raw_result': response_json
        }
        logger.info(f"Returning analysis result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            'depression_score': 0,
            'confidence': 0,
            'processed_text': text,
            'error': str(e)
        }


@login_required
def analyze_audio(request):
    logger = logging.getLogger(__name__)
    logger.info(f"analyze_audio called - method: {request.method}")
    
    if request.method == 'POST':
        logger.info("Processing POST request for audio analysis")
        try:
            # Validate session
            logger.info(f"Session keys: {list(request.session.keys())}")
            if 'phq9_data' not in request.session:
                logger.warning("No PHQ9 data in session, redirecting to phq9_view")
                return redirect('phq9_view')
            
            logger.info(f"PHQ9 data in session: {json.dumps(request.session['phq9_data'])}")
            
            # Get transcription from form
            transcription = request.POST.get('transcription', '').strip()
            logger.info(f"Transcription length: {len(transcription)}")
            if not transcription:
                logger.error("No transcription received")
                raise ValueError("No transcription received")
            
            # Analyze the transcribed text
            logger.info("Calling analyze_text_with_model")
            analysis = analyze_text_with_model(transcription)
            logger.info(f"Analysis result: {json.dumps(analysis)}")
            
            # Check if analysis failed
            if 'error' in analysis:
                logger.error(f"Text analysis error: {analysis['error']}")
                return render(request, 'app/error.html', {
                    'error': f"Analysis failed: {analysis['error']}"
                })
            
            # Store values securely before creating test result
            try:
                phq9_score = int(request.session['phq9_data'].get('form_score', 0))
                total_score = int(request.session['phq9_data'].get('total_score', 0))
                depression_status = str(request.session['phq9_data'].get('depression_status', ''))
                emotion_counts = request.session['phq9_data'].get('emotion_counts', {})
                emotion_score = int(request.session['phq9_data'].get('emotion_score', 0))
                
                logger.info(f"Extracted data: phq9_score={phq9_score}, total_score={total_score}, "
                           f"status={depression_status}, emotion_score={emotion_score}")
            except Exception as e:
                logger.error(f"Error extracting session data: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Invalid session data format: {str(e)}")
            
            # Create test result
            logger.info("Creating TestResult object")
            try:
                result = TestResult.objects.create(
                    user=request.user,
                    phq9_score=phq9_score,
                    total_score=total_score,
                    Status=depression_status,
                    emotions=emotion_counts,
                    emotion_score=emotion_score,
                    audio_analysis={
                        'depression_score': analysis.get('depression_score', 0),
                        'processed_text': analysis.get('processed_text', ''),
                        'confidence': analysis.get('confidence', 0)
                    },
                    audio_duration=20  # Fixed duration
                )
                
                # Check if the result was created successfully
                logger.info(f"TestResult created with ID: {result.id}")
                
                # Explicitly save the result
                result.save()
                logger.info(f"TestResult saved with ID: {result.id}")
                
                # Verify the object exists in the database
                verification = TestResult.objects.filter(id=result.id).exists()
                logger.info(f"TestResult verification check: {verification}")
                
            except Exception as db_error:
                logger.error(f"Database error creating TestResult: {str(db_error)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Failed to create test result: {str(db_error)}")
            
            # Clear session data after successful save
            logger.info("Clearing session data")
            try:
                del request.session['phq9_data']
                request.session.modified = True
                logger.info("Session data cleared successfully")
            except Exception as session_error:
                logger.error(f"Error clearing session: {str(session_error)}")
                # Continue even if clearing session fails
            
            # Construct redirect URL
            redirect_url = reverse('final_results', kwargs={'result_id': result.id})
            logger.info(f"Redirecting to: {redirect_url}")
            
            # Redirect to results page with the new result ID
            return redirect('final_results', result_id=result.id)

        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            logger.error(traceback.format_exc())
            return render(request, 'app/error.html', {
                'error': f"Processing failed: {str(e)}"
            })
    
    # If not a POST request
    logger.info("Not a POST request, redirecting to audio_phase")
    return redirect('audio_phase')

def calculate_composite_score(result):
    """Calculate weighted composite score with robust error handling"""
    try:
        # Safely get all scores with defaults
        phq9_score = float(getattr(result, 'phq9_score', 0))
        emotion_score = float(getattr(result, 'emotion_score', 0))
        
        # Handle audio_analysis safely
        audio_data = getattr(result, 'audio_analysis', {}) or {}  # Double safety
        audio_score = float(audio_data.get('depression_score', 0))
        
        # Define weights
        weights = {
            'phq9': 0.5,
            'emotion': 0.2,
            'audio': 0.3
        }
        
        # Calculate weighted score
        return (
            (phq9_score * weights['phq9']) +
            (emotion_score * weights['emotion']) +
            (audio_score * weights['audio'])
        )
    except Exception as e:
        logger.error(f"Error calculating composite score: {str(e)}")
        return 0  # Return default score if calculation fails




@login_required
def final_results(request, result_id):
    logger = logging.getLogger(__name__)
    logger.info(f"final_results called for result_id: {result_id}, user: {request.user.username}")
    
    try:
        # Try to fetch the result with detailed error handling
        logger.info(f"Attempting to get TestResult with ID: {result_id}")
        
        # Check if the record exists at all, regardless of user
        exists_check = TestResult.objects.filter(id=result_id).exists()
        logger.info(f"TestResult with ID {result_id} exists in database: {exists_check}")
        
        if exists_check:
            # Check if it belongs to the current user
            user_match = TestResult.objects.filter(id=result_id, user=request.user).exists()
            logger.info(f"TestResult belongs to current user: {user_match}")
        
        # Use get_object_or_404 to fetch the object
        result = get_object_or_404(TestResult, id=result_id, user=request.user)
        logger.info(f"TestResult found: {result.id}")
        
        # Log available attributes to verify structure
        logger.info(f"TestResult attributes: phq9_score={result.phq9_score}, "
                   f"status={result.Status}, audio_analysis exists: {hasattr(result, 'audio_analysis')}")
        
        # Safely prepare context data with robust error handling
        audio_data = {}
        audio_score = 0
        has_audio_data = False
        
        try:
            audio_data = getattr(result, 'audio_analysis', {}) or {}
            logger.info(f"Audio data: {audio_data}")
            
            if audio_data:
                audio_score = float(audio_data.get('depression_score', 0))
                has_audio_data = True
                logger.info(f"Audio score: {audio_score}")
        except Exception as audio_err:
            logger.error(f"Error processing audio data: {str(audio_err)}")
            # Continue with default values
        
        try:
            composite_score = calculate_composite_score(result)
            logger.info(f"Calculated composite score: {composite_score}")
        except Exception as score_err:
            logger.error(f"Error calculating composite score: {str(score_err)}")
            composite_score = 0  # Fallback value
        
        if composite_score >= 20:
            result_type = "Severe Depression"
        elif composite_score >= 15:
            result_type = "Moderately Severe Depression"
        elif composite_score >= 10:
            result_type = "Moderate Depression"
        elif composite_score >= 5:
            result_type = "Mild Depression"
        else:
            result_type = "Minimal or No Depression"
        
        recommendation=get_recommendation(composite_score, result_type)
        # Build context
        context = {
            'result': result,
            'composite_score': composite_score,
            'audio_data': audio_data,
            'has_audio_data': has_audio_data,
            'result_type': result_type,
            'recommendation': recommendation,
        }
        
        logger.info("Rendering final_result.html template")
        return render(request, 'app/result.html', context)
    
    except Http404:
        logger.error(f"TestResult with ID {result_id} not found")
        return render(request, 'app/error.html', {
            'error': "Could not load test results. Please try again."
        }, status=404)
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        logger.error(traceback.format_exc())
        return render(request, 'app/error.html', {
            'error': "Could not load test results. Please try again."
        }, status=500)
        
        


def analyze_journal_text(text):
    """Analyze text using Cloudflare's sentiment model"""
    API_KEY = os.getenv("CLOUDFLARE_API_TOKEN")
    ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    MODEL = "@cf/huggingface/distilbert-sst-2-int8"
    
    if not API_KEY or not ACCOUNT_ID:
        logger.error("Missing Cloudflare API credentials")
        return {'error': 'API configuration missing'}

    try:
        response = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{MODEL}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"text": text},
            timeout=10
        )
        
        if response.status_code != 200:
            return {'error': f'API error: {response.status_code}'}

        result = response.json().get('result', [])
        scores = {
            'positive': 0.0,
            'negative': 0.0
        }

        for item in result:
            if item['label'] == 'POSITIVE':
                scores['positive'] = float(item['score'])
            elif item['label'] == 'NEGATIVE':
                scores['negative'] = float(item['score'])

        return scores

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {'error': str(e)}

def journal(request):
    if request.method == 'POST':
        form = JournalForm(request.POST)
        if form.is_valid():
            entry = form.save(commit=False)
            entry.user = request.user
            
            # Perform sentiment analysis
            analysis = analyze_journal_text(entry.content)
            
            if 'error' not in analysis:
                entry.positive_score = analysis.get('positive', 0)
                entry.negative_score = analysis.get('negative', 0)
                entry.save()
                return redirect('journal')
            else:
                # Handle analysis error
                messages.error(request, f"Analysis failed: {analysis['error']}")

    entries = JournalEntry.objects.filter(user=request.user).order_by('-entry_date')
    return render(request, 'app/journal.html', {
        'form': JournalForm(),
        'entries': entries
    })