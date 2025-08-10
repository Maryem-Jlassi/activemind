from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import os
import json
import uuid
import asyncio
import tempfile
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import threading
import time
import requests
from io import BytesIO

# Import your existing classes (assuming they're in separate modules)
from logic import ConversationalVoiceCoach, AthleteProfile, SportType, AthleteLevel, Config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# Configuration
class Config:
    MODEL_NAME = "ollama/llama3.2:latest"
    API_BASE = "http://localhost:11434"
    SESSION_DURATION_MINUTES = 20
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "key_api")
    ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam voice
    TTS_LANGUAGE = "en"
    DATABASE_PATH = "coach_sessions.db"
    MAX_RESPONSE_LENGTH = 50

class AthleteLevel(Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    ELITE = "Elite"

class SportType(Enum):
    ENDURANCE = "Endurance Sports"
    STRENGTH = "Strength Training"
    TEAM_SPORTS = "Team Sports"
    INDIVIDUAL = "Individual Sports"
    COMBAT = "Combat Sports"
    MIXED = "Mixed Training"

@dataclass
class AthleteProfile:
    session_id: str
    name: str = ""
    age: int = 0
    weight: float = 0.0
    height: float = 0.0
    sport_type: Optional[SportType] = None
    athlete_level: Optional[AthleteLevel] = None
    training_frequency: int = 0
    primary_goals: List[str] = field(default_factory=list)
    current_responses: Dict[str, str] = field(default_factory=dict)
    consultation_history: List[Dict] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    data_collection_complete: bool = False

# Import your existing voice coach logic here
from crewai import Agent, Task, Crew, Process, LLM
import speech_recognition as sr

class ElevenLabsVoiceService:
    def __init__(self):
        self.api_key = Config.ELEVENLABS_API_KEY
        self.voice_id = Config.ELEVENLABS_VOICE_ID
        self.recognizer = sr.Recognizer()
        self.base_url = "https://api.elevenlabs.io/v1"
        
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
    
    def text_to_speech_elevenlabs(self, text: str) -> Optional[bytes]:
        if not self.api_key or self.api_key == "key_api":
            return None
            
        try:
            clean_text = self._prepare_text_for_speech(text)
            
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": clean_text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.6,
                    "similarity_boost": 0.8,
                    "style": 0.4,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"ElevenLabs API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")
            return None
    
    def _prepare_text_for_speech(self, text: str) -> str:
        text = text.replace('**', '')
        text = text.replace('*', '')
        text = text.replace('  ', ' ')
        text = text.replace('. ', '... ')
        
        words = text.split()
        if len(words) > Config.MAX_RESPONSE_LENGTH:
            text = ' '.join(words[:Config.MAX_RESPONSE_LENGTH]) + '...'
        
        return text.strip()

class ConversationalVoiceCoach:
    def __init__(self):
        self.llm = LLM(model=Config.MODEL_NAME, base_url=Config.API_BASE)
        self.voice_service = ElevenLabsVoiceService()
        
        self.data_collection_steps = [
            "name", "age", "sport", "level", "training_frequency", 
            "goals", "current_feeling", "confirmation"
        ]
    
    def create_conversational_agent(self, context: str = "") -> Agent:
        return Agent(
            role='Coach Alex - Conversational Mental Performance Coach',
            goal='Have natural, brief conversations like a supportive mentor',
            backstory=f'''You are Coach Alex, a confident and supportive male mental performance coach who talks 
            like a knowledgeable friend having a casual conversation. You keep your responses SHORT and conversational.
            
            KEY COMMUNICATION STYLE:
            - Maximum 2-3 sentences per response
            - Confident, encouraging tone like a trusted mentor
            - Ask ONE question at a time
            - Use conversational language ("Great!", "Excellent!", "I see")
            - Show genuine interest but keep it brief
            - Use the person's name occasionally, not every response
            - React naturally to what they say
            - Sound like a male coach with experience and confidence
            
            CURRENT CONTEXT:
            {context}
            
            IMPORTANT: Keep responses under 30 words for natural conversation flow.
            ''',
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )
    
    def generate_coaching_question(self, profile: AthleteProfile, question_context: str = "") -> Dict[str, Any]:
        agent = self.create_conversational_agent(f"Athlete: {profile.name}, {profile.age} years old")
        
        task = Task(
            description=f'''
            As Coach Alex, ask ONE short, conversational coaching question about {profile.name}'s mental game.
            
            ATHLETE INFO:
            - Name: {profile.name}
            - Age: {profile.age}
            - Sport: {profile.sport_type.value if profile.sport_type else 'General'}
            - Level: {profile.athlete_level.value if profile.athlete_level else 'Intermediate'}
            
            Ask ONE specific question about mental performance.
            Keep it SHORT and conversational!
            ''',
            expected_output='One short, conversational coaching question',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        question_text = str(result).strip()
        audio_data = self.generate_voice_audio(question_text)
        
        return {
            'question': question_text,
            'audio': audio_data
        }
    
    def generate_coaching_response(self, profile: AthleteProfile, user_response: str) -> Dict[str, Any]:
        agent = self.create_conversational_agent()
        
        task = Task(
            description=f'''
            As Coach Alex, respond to {profile.name}'s answer: "{user_response}"
            
            YOUR RESPONSE SHOULD:
            - Acknowledge what they said (1 sentence)
            - Provide ONE brief insight or tip (1 sentence)
            - Be encouraging and confident
            - Keep it under 25 words total
            
            Be conversational, brief, and confident!
            ''',
            expected_output='Brief, supportive coaching response under 25 words',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        response_text = str(result).strip()
        audio_data = self.generate_voice_audio(response_text)
        
        return {
            'response': response_text,
            'audio': audio_data
        }
    
    def generate_voice_audio(self, text: str) -> Optional[str]:
        try:
            audio_data = self.voice_service.text_to_speech_elevenlabs(text)
            
            if audio_data:
                # Convert to base64 for WebSocket transmission
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                return audio_base64
            else:
                return None
                
        except Exception as e:
            print(f"Voice generation error: {e}")
            return None

# Global instances
voice_coach = ConversationalVoiceCoach()
active_sessions = {}

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@socketio.on('connect')
def handle_connect():
    session_id = str(uuid.uuid4())
    join_room(session_id)
    
    # Create new athlete profile
    profile = AthleteProfile(session_id=session_id)
    active_sessions[session_id] = {
        'profile': profile,
        'conversation_history': [],
        'current_step': 'welcome'
    }
    
    emit('session_created', {
        'session_id': session_id,
        'message': "Connected to Coach Alex!"
    })
    
    print(f"Client connected: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('start_conversation')
def handle_start_conversation(data):
    session_id = data.get('session_id')
    if session_id not in active_sessions:
        emit('error', {'message': 'Invalid session'})
        return
    
    # Send welcome message
    welcome_message = "Hey there! I'm Coach Alex, your mental performance coach. What should I call you?"
    audio_data = voice_coach.generate_voice_audio(welcome_message)
    
    emit('coach_message', {
        'message': welcome_message,
        'audio': audio_data,
        'animation': 'Greeting',
        'step': 'name'
    })

@socketio.on('user_response')
def handle_user_response(data):
    session_id = data.get('session_id')
    user_input = data.get('message', '')
    current_step = data.get('step', 'welcome')
    
    if session_id not in active_sessions:
        emit('error', {'message': 'Invalid session'})
        return
    
    session_data = active_sessions[session_id]
    profile = session_data['profile']
    
    # Update profile based on input
    update_profile_from_input(profile, user_input, current_step)
    
    # Generate coach response
    try:
        if current_step in ['name', 'age', 'sport', 'level', 'training_frequency', 'goals']:
            # Data collection phase
            next_step, response_text = get_next_conversation_step(profile, current_step)
            audio_data = voice_coach.generate_voice_audio(response_text)
            animation = 'Greeting' if current_step == 'name' else 'Idle'
            
            emit('coach_message', {
                'message': response_text,
                'audio': audio_data,
                'animation': animation,
                'step': next_step
            })
            
        else:
            # Coaching phase
            coaching_response = voice_coach.generate_coaching_response(profile, user_input)
            
            emit('coach_message', {
                'message': coaching_response['response'],
                'audio': coaching_response['audio'],
                'animation': 'Idle',
                'step': 'coaching'
            })
        
        # Update conversation history
        session_data['conversation_history'].append({
            'user_message': user_input,
            'coach_response': response_text if 'response_text' in locals() else coaching_response.get('response', ''),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error generating response: {e}")
        emit('error', {'message': 'Error generating response'})

@socketio.on('get_coaching_question')
def handle_get_coaching_question(data):
    session_id = data.get('session_id')
    
    if session_id not in active_sessions:
        emit('error', {'message': 'Invalid session'})
        return
    
    session_data = active_sessions[session_id]
    profile = session_data['profile']
    
    try:
        question_data = voice_coach.generate_coaching_question(profile)
        
        emit('coach_message', {
            'message': question_data['question'],
            'audio': question_data['audio'],
            'animation': 'Greeting',
            'step': 'coaching'
        })
        
    except Exception as e:
        print(f"Error generating question: {e}")
        emit('error', {'message': 'Error generating question'})

def update_profile_from_input(profile: AthleteProfile, user_input: str, step: str):
    try:
        if step == 'name':
            name = user_input.split()[-1] if user_input.split() else user_input
            profile.name = name.strip().title()
            
        elif step == 'age':
            import re
            age_match = re.search(r'\d+', user_input)
            if age_match:
                profile.age = int(age_match.group())
                
        elif step == 'sport':
            sport_keywords = {
                SportType.ENDURANCE: ['running', 'cycling', 'swimming', 'marathon', 'triathlon'],
                SportType.STRENGTH: ['weightlifting', 'powerlifting', 'strength', 'gym', 'lifting'],
                SportType.TEAM_SPORTS: ['football', 'basketball', 'soccer', 'volleyball', 'hockey'],
                SportType.INDIVIDUAL: ['tennis', 'golf', 'track', 'individual'],
                SportType.COMBAT: ['boxing', 'mma', 'wrestling', 'martial arts'],
                SportType.MIXED: ['crossfit', 'mixed', 'cross training', 'general fitness']
            }
            
            user_lower = user_input.lower()
            for sport_type, keywords in sport_keywords.items():
                if any(keyword in user_lower for keyword in keywords):
                    profile.sport_type = sport_type
                    break
            else:
                profile.sport_type = SportType.MIXED
                
        elif step == 'level':
            user_lower = user_input.lower()
            if any(word in user_lower for word in ['beginner', 'new', 'started']):
                profile.athlete_level = AthleteLevel.BEGINNER
            elif any(word in user_lower for word in ['intermediate', 'year', 'years']):
                profile.athlete_level = AthleteLevel.INTERMEDIATE
            elif any(word in user_lower for word in ['advanced', 'experienced']):
                profile.athlete_level = AthleteLevel.ADVANCED
            else:
                profile.athlete_level = AthleteLevel.INTERMEDIATE
                
    except Exception as e:
        print(f"Error updating profile: {e}")

def get_next_conversation_step(profile: AthleteProfile, current_step: str) -> tuple[str, str]:
    step_responses = {
        'name': (
            'age',
            f"Great to meet you, {profile.name}! How old are you?"
        ),
        'age': (
            'sport',
            f"Perfect! What sport do you focus on, {profile.name}?"
        ),
        'sport': (
            'level',
            f"Awesome! How would you describe your experience level?"
        ),
        'level': (
            'training_frequency', 
            f"Excellent! How many times do you typically train per week?"
        ),
        'training_frequency': (
            'goals',
            f"Got it! What's your main goal you'd like to work on today?"
        ),
        'goals': (
            'coaching',
            f"That's a solid goal! Let's dive into your mental performance session."
        )
    }
    
    return step_responses.get(current_step, ('coaching', "Let's continue with your coaching session!"))

if __name__ == '__main__':
    print("Starting Coach Alex Voice Backend...")
    print("Make sure you have:")
    print("1. Ollama running with llama3.2")
    print("2. ElevenLabs API key set")
    print("3. Required Python packages installed")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
