import os
import json
import uuid
import sqlite3
import tempfile
import base64
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import speech_recognition as sr
from crewai import Agent, Task, Crew, Process, LLM
import requests
from io import BytesIO
import threading
import time

# Configuration
class Config:
    MODEL_NAME = "ollama/llama3.2:latest"
    API_BASE = "http://localhost:11434"
    SESSION_DURATION_MINUTES = 20
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_b0b1edcc7ae4ae43503cb6f52f3c4768b8f6a70c4f82ee32")
    ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam voice - confident male voice
    TTS_LANGUAGE = "en"
    DATABASE_PATH = "coach_sessions.db"
    MAX_RESPONSE_LENGTH = 50
    UPLOAD_FOLDER = "temp_audio"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

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

class ElevenLabsVoiceService:
    """ElevenLabs TTS and Speech Recognition service"""
    
    def __init__(self):
        self.api_key = Config.ELEVENLABS_API_KEY
        self.voice_id = Config.ELEVENLABS_VOICE_ID
        self.recognizer = sr.Recognizer()
        self.base_url = "https://api.elevenlabs.io/v1"
        
        # Adjust recognizer settings for better performance
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
    def text_to_speech_elevenlabs(self, text: str) -> Optional[bytes]:
        """Generate speech using ElevenLabs API"""
        if not self.api_key or self.api_key == "your_elevenlabs_api_key_here":
            print("⚠️ ElevenLabs API key not configured. Using fallback TTS.")
            return None
            
        try:
            # Clean and prepare text for speech
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
        """Prepare text for natural speech"""
        # Remove excessive punctuation and clean up
        text = text.replace('**', '')
        text = text.replace('*', '')
        text = text.replace('  ', ' ')
        
        # Add natural pauses for better speech flow
        text = text.replace('. ', '... ')
        text = text.replace('!', '!')
        text = text.replace('?', '?')
        
        # Limit length for conversational flow
        words = text.split()
        if len(words) > Config.MAX_RESPONSE_LENGTH:
            text = ' '.join(words[:Config.MAX_RESPONSE_LENGTH]) + '...'
        
        return text.strip()
    
    def speech_to_text(self, audio_data: bytes, audio_format: str = "webm") -> Optional[str]:
        """Convert audio to text using speech recognition"""
        try:
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Use speech recognition with the audio file
                with sr.AudioFile(temp_file_path) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio)
                    return text
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return None

class DatabaseManager:
    """Manages athlete session persistence"""
    
    def __init__(self, db_path: str = Config.DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS athlete_sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                weight REAL,
                height REAL,
                sport_type TEXT,
                athlete_level TEXT,
                training_frequency INTEGER,
                primary_goals TEXT,
                current_responses TEXT,
                consultation_history TEXT,
                preferences TEXT,
                data_collection_complete BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consultation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                phase TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES athlete_sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_athlete_profile(self, profile: AthleteProfile):
        """Save or update athlete profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO athlete_sessions 
                (session_id, name, age, weight, height, sport_type, athlete_level, 
                 training_frequency, primary_goals, current_responses, consultation_history, 
                 preferences, data_collection_complete, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.session_id,
                profile.name,
                profile.age,
                profile.weight,
                profile.height,
                profile.sport_type.value if profile.sport_type else "",
                profile.athlete_level.value if profile.athlete_level else "",
                profile.training_frequency,
                json.dumps(profile.primary_goals),
                json.dumps(profile.current_responses),
                json.dumps(profile.consultation_history),
                json.dumps(profile.preferences),
                profile.data_collection_complete,
                profile.timestamp.isoformat(),
                datetime.now().isoformat()
            ))
            
            conn.commit()
        finally:
            conn.close()
    
    def load_athlete_profile(self, session_id: str) -> Optional[AthleteProfile]:
        """Load athlete profile by session ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM athlete_sessions WHERE session_id = ?', (session_id,))
            row = cursor.fetchone()
            
            if row:
                return AthleteProfile(
                    session_id=row[0],
                    name=row[1] or "",
                    age=row[2] or 0,
                    weight=row[3] or 0.0,
                    height=row[4] or 0.0,
                    sport_type=SportType(row[5]) if row[5] else None,
                    athlete_level=AthleteLevel(row[6]) if row[6] else None,
                    training_frequency=row[7] or 0,
                    primary_goals=json.loads(row[8]) if row[8] else [],
                    current_responses=json.loads(row[9]) if row[9] else {},
                    consultation_history=json.loads(row[10]) if row[10] else [],
                    preferences=json.loads(row[11]) if row[11] else {},
                    data_collection_complete=bool(row[12]),
                    timestamp=datetime.fromisoformat(row[13]) if row[13] else datetime.now()
                )
        except Exception as e:
            print(f"Error loading profile: {e}")
            return None
        finally:
            conn.close()
        
        return None

class ConversationalVoiceCoach:
    """Voice-first conversational mental performance coach"""
    
    def __init__(self):
        self.llm = LLM(model=Config.MODEL_NAME, base_url=Config.API_BASE)
        self.voice_service = ElevenLabsVoiceService()
        self.db_manager = DatabaseManager()
        
        # Conversation flow steps for data collection
        self.data_collection_steps = [
            "name", "age", "sport", "level", "training_frequency", 
            "goals", "current_feeling", "confirmation", "complete"
        ]
    
    def create_conversational_agent(self, context: str = "") -> Agent:
        """Creates a conversational coach agent with short responses"""
        
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
            This is like texting or instant messaging - short, confident, and to the point.
            ''',
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )
    
    def start_vocal_data_collection(self, profile: AthleteProfile) -> Dict[str, Any]:
        """Start the vocal data collection conversation"""
        
        welcome_text = "Hey there! I'm Coach Alex, your mental performance coach. What should I call you?"
        
        audio_file = self.generate_voice_audio(welcome_text)
        
        self.db_manager.save_athlete_profile(profile)
        
        return {
            'message': welcome_text,
            'audio_url': f'/media/{os.path.basename(audio_file)}' if audio_file else None,
            'step': 'name',
            'expecting': 'user_name'
        }
    
    def process_conversation_step(self, profile: AthleteProfile, user_input: str, step: str) -> Dict[str, Any]:
        """Process each step of the conversational data collection"""
        
        # Update profile based on current step
        if user_input.strip():
            self._update_profile_from_input(profile, user_input, step)
        
        # Generate next conversation step
        next_step, response_text = self._get_next_conversation_step(profile, user_input, step)
        
        # Generate audio response
        audio_file = self.generate_voice_audio(response_text)
        
        # Update profile
        self.db_manager.save_athlete_profile(profile)
        
        return {
            'message': response_text,
            'audio_url': f'/media/{os.path.basename(audio_file)}' if audio_file else None,
            'step': next_step,
            'profile_updated': True
        }
    
    def _update_profile_from_input(self, profile: AthleteProfile, user_input: str, step: str):
        """Update profile based on user input and current step"""
        
        try:
            if step == 'name':
                # Extract name from input
                name = user_input.split()[-1] if user_input.split() else user_input
                profile.name = name.strip().title()
                
            elif step == 'age':
                # Extract age
                import re
                age_match = re.search(r'\d+', user_input)
                if age_match:
                    profile.age = int(age_match.group())
                    
            elif step == 'sport':
                # Determine sport type from input
                sport_keywords = {
                    SportType.ENDURANCE: ['running', 'cycling', 'swimming', 'marathon', 'triathlon', 'endurance'],
                    SportType.STRENGTH: ['weightlifting', 'powerlifting', 'strength', 'gym', 'lifting'],
                    SportType.TEAM_SPORTS: ['football', 'basketball', 'soccer', 'volleyball', 'hockey', 'team'],
                    SportType.INDIVIDUAL: ['tennis', 'golf', 'track', 'individual'],
                    SportType.COMBAT: ['boxing', 'mma', 'wrestling', 'martial arts', 'fighting'],
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
                # Determine level from input
                user_lower = user_input.lower()
                if any(word in user_lower for word in ['beginner', 'new', 'started', 'beginning']):
                    profile.athlete_level = AthleteLevel.BEGINNER
                elif any(word in user_lower for word in ['intermediate', 'year', 'years', 'some experience']):
                    profile.athlete_level = AthleteLevel.INTERMEDIATE
                elif any(word in user_lower for word in ['advanced', 'experienced', 'competitive']):
                    profile.athlete_level = AthleteLevel.ADVANCED
                elif any(word in user_lower for word in ['elite', 'professional', 'pro']):
                    profile.athlete_level = AthleteLevel.ELITE
                else:
                    profile.athlete_level = AthleteLevel.INTERMEDIATE
                    
            elif step == 'training_frequency':
                # Extract training frequency
                import re
                freq_match = re.search(r'\d+', user_input)
                if freq_match:
                    profile.training_frequency = int(freq_match.group())
                else:
                    profile.training_frequency = 3
                    
            elif step == 'goals':
                # Add goals
                if user_input.strip():
                    profile.primary_goals.append(user_input.strip())
                    
        except Exception as e:
            print(f"Error updating profile: {e}")
    
    def _get_next_conversation_step(self, profile: AthleteProfile, user_input: str, step: str) -> tuple[str, str]:
        """Get the next conversation step and response"""
        
        step_responses = {
            'welcome': (
                'name',
                "Hey there! I'm Coach Alex, your mental performance coach. What should I call you?"
            ),
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
                f"Awesome! How would you describe your experience level in {profile.sport_type.value.lower() if profile.sport_type else 'your sport'}?"
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
                'current_feeling',
                f"That's a solid goal! How are you feeling about your performance lately?"
            ),
            'current_feeling': (
                'confirmation',
                f"Thanks for sharing that! Let me quickly confirm what I've got..."
            ),
            'confirmation': (
                'complete',
                self._generate_confirmation_message(profile)
            ),
            'complete': (
                'complete',
                "All set! We're ready to start your coaching session."
            )
        }
        
        return step_responses.get(step, ('complete', "All set! We're ready to start your coaching session."))
    
    def _generate_confirmation_message(self, profile: AthleteProfile) -> str:
        """Generate confirmation message with collected data"""
        return (f"So you're {profile.name}, {profile.age} years old, doing {profile.sport_type.value.lower() if profile.sport_type else 'sports'} "
                f"at {profile.athlete_level.value.lower() if profile.athlete_level else 'intermediate'} level, "
                f"training {profile.training_frequency} times a week. Sound right?")
    
    def generate_coaching_question(self, profile: AthleteProfile, question_context: str = "") -> Dict[str, Any]:
        """Generate a single coaching question"""
        agent = self.create_conversational_agent(f"Athlete: {profile.name}, {profile.age} years old, {profile.sport_type.value if profile.sport_type else 'athlete'}")
        
        task = Task(
            description=f'''
            As Coach Alex, ask ONE short, conversational coaching question about {profile.name}'s mental game.
            
            ATHLETE INFO:
            - Name: {profile.name}
            - Age: {profile.age}
            - Sport: {profile.sport_type.value if profile.sport_type else 'General'}
            - Level: {profile.athlete_level.value if profile.athlete_level else 'Intermediate'}
            - Training: {profile.training_frequency} times/week
            - Goals: {', '.join(profile.primary_goals)}
            
            CONTEXT: {question_context}
            
            REQUIREMENTS:
            - Ask ONE specific question 
            - Make it conversational and natural
            - Focus on mental performance aspects
            - Be confident and encouraging
            - Don't give advice yet, just ask
            - Sound like a confident male coach
            
            TOPICS TO EXPLORE:
            - Confidence levels
            - Pre-competition nerves
            - Focus during training
            - Motivation patterns
            - Stress management
            - Recovery mindset
            
            Example: "How confident do you feel going into competitions lately?"
            
            Keep it SHORT and conversational!
            ''',
            expected_output='One short, conversational coaching question',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        question_text = str(result).strip()
        audio_file = self.generate_voice_audio(question_text)
        
        return {
            'question': question_text,
            'audio_url': f'/media/{os.path.basename(audio_file)}' if audio_file else None
        }
    
    def generate_coaching_response(self, profile: AthleteProfile, user_response: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Generate a coaching response to user's answer"""
        
        context = self._build_conversation_context(profile, conversation_history)
        
        agent = self.create_conversational_agent(context)
        
        task = Task(
            description=f'''
            As Coach Alex, respond to {profile.name}'s answer in a brief, supportive way.
            
            THEIR RESPONSE: "{user_response}"
            
            YOUR RESPONSE SHOULD:
            - Acknowledge what they said (1 sentence)
            - Provide ONE brief insight or tip (1 sentence)
            - Be encouraging and confident
            - Keep it under 25 words total
            - Sound like a knowledgeable male coach giving advice
            
            EXAMPLES:
            - "I hear you! Try focusing on your breathing when that happens."
            - "That's normal. Remember, nerves show you care - channel that energy!"
            - "Good awareness! Quick visualization before training can help with that."
            
            Be conversational, brief, and confident!
            ''',
            expected_output='Brief, supportive coaching response under 25 words',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        response_text = str(result).strip()
        audio_file = self.generate_voice_audio(response_text)
        
        return {
            'response': response_text,
            'audio_url': f'/media/{os.path.basename(audio_file)}' if audio_file else None
        }
    
    def generate_final_coaching_summary(self, profile: AthleteProfile, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Generate final coaching summary"""
        
        context = self._build_conversation_context(profile, conversation_history)
        
        agent = self.create_conversational_agent(context)
        
        task = Task(
            description=f'''
            As Coach Alex, provide a final summary and action plan for {profile.name}.
            
            CONVERSATION SUMMARY:
            {context}
            
            PROVIDE:
            1. Brief acknowledgment (1 sentence)
            2. Key insight from our conversation (1 sentence)
            3. Some specific techniques to practice 
            4. Confident closing (1 sentence)
            5. Make it short 
            
            TOTAL: Keep it actionable and memorable!
            Sound like a confident male coach.
            
            Example structure:
            "Great session, {profile.name}! I can see your dedication clearly. 
            Try the 3-breath confidence technique before training - breathe in strength, hold focus, breathe out doubt. 
            You've got this champion!"
            ''',
            expected_output='Brief, actionable coaching summary under 50 words',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        summary_text = str(result).strip()
        audio_file = self.generate_voice_audio(summary_text)
        
        return {
            'summary': summary_text,
            'audio_url': f'/media/{os.path.basename(audio_file)}' if audio_file else None
        }
    
    def _build_conversation_context(self, profile: AthleteProfile, history: List[Dict]) -> str:
        """Build context from conversation history"""
        context = f"Athlete: {profile.name}, {profile.age}, {profile.sport_type.value if profile.sport_type else 'athlete'}\n"
        context += f"Goals: {', '.join(profile.primary_goals)}\n\n"
        context += "Conversation:\n"
        
        for exchange in history[-5:]:  # Last 5 exchanges
            context += f"Q: {exchange.get('question', exchange.get('coach_message', ''))}\n"
            context += f"A: {exchange.get('response', exchange.get('user_response', ''))}\n"
        
        return context
    
    def generate_voice_audio(self, text: str, filename_prefix: str = "coach") -> Optional[str]:
        """Generate voice audio using ElevenLabs"""
        try:
            audio_data = self.voice_service.text_to_speech_elevenlabs(text)
            
            if audio_data:
                # Ensure upload folder exists
                os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
                
                # Save to file
                filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.mp3"
                file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                
                with open(file_path, 'wb') as f:
                    f.write(audio_data)
                
                return file_path
            else:
                print("ElevenLabs not available, no audio generated")
                return None
                
        except Exception as e:
            print(f"Voice generation error: {e}")
            return None

# Flask Application
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:5173"])  # Allow React dev server

app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

# Initialize services
coach_service = ConversationalVoiceCoach()

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/voice/session', methods=['POST'])
def create_session():
    """Create new voice coaching session"""
    try:
        # Get query parameters
        tts_enabled = request.args.get('tts', 'true').lower() == 'true'
        voice_name = request.args.get('voice', 'Rachel')
        
        # Get profile data from request
        profile_data = request.json
        
        # Create new profile
        session_id = str(uuid.uuid4())
        profile = AthleteProfile(session_id=session_id)
        
        # Update profile with provided data
        if profile_data:
            profile.name = profile_data.get('name', '')
            profile.age = profile_data.get('age', 0)
            profile.weight = profile_data.get('weight', 0.0)
            profile.height = profile_data.get('height', 0.0)
            profile.training_frequency = profile_data.get('training_frequency', 0)
            
            # Handle sport type
            sport_type_str = profile_data.get('sport_type', '')
            if sport_type_str:
                try:
                    profile.sport_type = SportType(sport_type_str)
                except ValueError:
                    profile.sport_type = SportType.MIXED
            
            # Handle athlete level
            athlete_level_str = profile_data.get('athlete_level', '')
            if athlete_level_str:
                try:
                    profile.athlete_level = AthleteLevel(athlete_level_str)
                except ValueError:
                    profile.athlete_level = AthleteLevel.INTERMEDIATE
            
            # Handle goals
            goals = profile_data.get('goals', [])
            if isinstance(goals, list):
                profile.primary_goals = goals
            
            # Handle preferences
            preferences = profile_data.get('preferences', {})
            if isinstance(preferences, dict):
                profile.preferences = preferences
        
        # Save profile
        coach_service.db_manager.save_athlete_profile(profile)
        
        # Start data collection
        result = coach_service.start_vocal_data_collection(profile)
        
        return jsonify({
            "sessionId": session_id,
            "next": result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voice/step', methods=['POST'])
def process_step():
    """Process a conversation step"""
    try:
        # Get query parameters
        tts_enabled = request.args.get('tts', 'true').lower() == 'true'
        voice_name = request.args.get('voice', 'Rachel')
        
        # Get request data
        data = request.json
        session_id = data.get('session_id')
        step = data.get('step')
        user_input = data.get('user_input', '')
        
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        
        # Load profile
        profile = coach_service.db_manager.load_athlete_profile(session_id)
        if not profile:
            return jsonify({"error": "Session not found"}), 404
        
        # Process the conversation step
        result = coach_service.process_conversation_step(profile, user_input, step)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voice/question', methods=['POST'])
def ask_question():
    """Generate a coaching question"""
    try:
        # Get query parameters
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        
        # Load profile
        profile = coach_service.db_manager.load_athlete_profile(session_id)
        if not profile:
            return jsonify({"error": "Session not found"}), 404
        
        # Generate coaching question
        result = coach_service.generate_coaching_question(profile)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voice/respond', methods=['POST'])
def respond_to_message():
    """Generate coaching response to user message"""
    try:
        # Get request data
        data = request.json
        session_id = data.get('session_id')
        user_message = data.get('user_message', '')
        history = data.get('history', [])
        
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        
        # Load profile
        profile = coach_service.db_manager.load_athlete_profile(session_id)
        if not profile:
            return jsonify({"error": "Session not found"}), 404
        
        # Generate coaching response
        result = coach_service.generate_coaching_response(profile, user_message, history)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voice/summary', methods=['POST'])
def generate_summary():
    """Generate session summary"""
    try:
        # Get query parameters
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        
        # Load profile
        profile = coach_service.db_manager.load_athlete_profile(session_id)
        if not profile:
            return jsonify({"error": "Session not found"}), 404
        
        # Get conversation history from profile
        history = profile.consultation_history
        
        # Generate summary
        result = coach_service.generate_final_coaching_summary(profile, history)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voice/utterance', methods=['POST'])
def process_audio():
    """Process audio utterance (speech-to-text)"""
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Determine file format
        file_format = "webm"  # Default for web recordings
        if audio_file.filename:
            file_format = audio_file.filename.split('.')[-1].lower()
        
        # Convert speech to text
        text = coach_service.voice_service.speech_to_text(audio_data, file_format)
        
        if text:
            return jsonify({"text": text})
        else:
            return jsonify({"error": "Could not process audio"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/assessment/submit', methods=['POST'])
def submit_assessment():
    """Submit performance assessment"""
    try:
        # Get query parameters
        tts_enabled = request.args.get('tts', 'true').lower() == 'true'
        voice_name = request.args.get('voice', 'Rachel')
        
        # Get request data
        data = request.json
        session_id = data.get('session_id')
        answers = data.get('answers', {})
        
        # Load profile if session exists
        profile = None
        if session_id:
            profile = coach_service.db_manager.load_athlete_profile(session_id)
        
        # Calculate assessment score
        score = calculate_assessment_score(answers)
        
        # Generate assessment summary
        summary = generate_assessment_summary(score, answers)
        
        # Generate training plan
        plan = generate_training_plan(score, answers, profile)
        
        # Generate audio if TTS enabled
        audio_file = None
        if tts_enabled:
            audio_text = f"{summary} Here's your plan: {'. '.join(plan[:3])}"
            audio_file = coach_service.generate_voice_audio(audio_text, "assessment")
        
        result = {
            "summary": summary,
            "score": score,
            "plan": plan,
            "audio_url": f'/media/{os.path.basename(audio_file)}' if audio_file else None
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/media/<filename>')
def serve_media(filename):
    """Serve audio files"""
    try:
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='audio/mpeg')
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session information"""
    try:
        profile = coach_service.db_manager.load_athlete_profile(session_id)
        if not profile:
            return jsonify({"error": "Session not found"}), 404
        
        # Convert profile to dict for JSON response
        profile_dict = {
            "session_id": profile.session_id,
            "nameunion_id": profile.session_id,
            "name": profile.name,
            "age": profile.age,
            "weight": profile.weight,
            "height": profile.height,
            "sport_type": profile.sport_type.value if profile.sport_type else None,
            "athlete_level": profile.athlete_level.value if profile.athlete_level else None,
            "training_frequency": profile.training_frequency,
            "primary_goals": profile.primary_goals,
            "current_responses": profile.current_responses,
            "consultation_history": profile.consultation_history,
            "preferences": profile.preferences,
            "data_collection_complete": profile.data_collection_complete,
            "timestamp": profile.timestamp.isoformat() if hasattr(profile.timestamp, 'isoformat') else str(profile.timestamp)
        }
        
        return jsonify(profile_dict)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List all sessions"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, name, sport_type, athlete_level, created_at, last_updated 
            FROM athlete_sessions 
            ORDER BY last_updated DESC
        ''')
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "session_id": row[0],
                "name": row[1],
                "sport_type": row[2],
                "athlete_level": row[3],
                "created_at": row[4],
                "last_updated": row[5]
            })
        
        conn.close()
        
        return jsonify({"sessions": sessions})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Assessment scoring functions
def calculate_assessment_score(answers: Dict[str, Any]) -> float:
    """Calculate assessment score from answers"""
    try:
        weights = {
            "execution": 0.4,
            "mood": 0.3,
            "fatigue": 0.3
        }
        
        # Define scoring scales
        execution_scale = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]
        mood_scale = ["Very Low", "Low", "Neutral", "Good", "Great"]
        fatigue_scale = ["None", "Mild", "Moderate", "High", "Extreme"]
        
        def scale_score(scale, value):
            try:
                index = scale.index(value)
                return (index + 1) * (100 / len(scale))
            except (ValueError, TypeError):
                return 50  # Default middle score
        
        # Calculate component scores
        execution_score = scale_score(execution_scale, answers.get("execution", "Sometimes"))
        mood_score = scale_score(mood_scale, answers.get("mood", "Neutral"))
        fatigue_score = scale_score(fatigue_scale, answers.get("fatigue", "Moderate"))
        
        # Invert fatigue score (less fatigue = higher score)
        fatigue_score = 100 - fatigue_score
        
        # Calculate weighted total
        total_score = (
            execution_score * weights["execution"] +
            mood_score * weights["mood"] +
            fatigue_score * weights["fatigue"]
        )
        
        return round(total_score, 1)
        
    except Exception as e:
        print(f"Assessment scoring error: {e}")
        return 50.0  # Default score

def generate_assessment_summary(score: float, answers: Dict[str, Any]) -> str:
    """Generate assessment summary based on score"""
    try:
        quick_state = answers.get("quick_state", "")
        
        if score >= 75:
            base_summary = "Peak performance state! Maintain current training intensity with focus on recovery."
        elif score >= 60:
            base_summary = "Good readiness detected. Continue with planned training while monitoring fatigue."
        elif score >= 40:
            base_summary = "Moderate readiness. Consider active recovery and mobility work today."
        else:
            base_summary = "Low readiness detected. Prioritize rest and hydration for optimal recovery."
        
        # Add state-specific insights
        if quick_state:
            if "Peak" in quick_state:
                base_summary += " You're in excellent form - capitalize on this peak state!"
            elif "Fatigued" in quick_state or "Overtrained" in quick_state:
                base_summary += " Your body is signaling for rest - listen to these important cues."
        
        return base_summary
        
    except Exception as e:
        return "Assessment complete. Focus on balanced training and recovery."

def generate_training_plan(score: float, answers: Dict[str, Any], profile: Optional[AthleteProfile] = None) -> List[str]:
    """Generate training plan based on assessment"""
    try:
        plan = []
        
        # Base recommendations based on score
        if score >= 75:
            plan.extend([
                "Warm-up: Dynamic stretching + sport-specific movements (10 min)",
                "Main: High-intensity training at 85-95% effort",
                "Focus: Skill refinement and peak performance techniques",
                "Cooldown: Active recovery + stretching (15 min)"
            ])
        elif score >= 60:
            plan.extend([
                "Warm-up: Light cardio + dynamic stretching (10 min)",
                "Main: Moderate intensity training at 70-85% effort",
                "Focus: Technical work and endurance building",
                "Cooldown: Static stretching + foam rolling (15 min)"
            ])
        elif score >= 40:
            plan.extend([
                "Warm-up: Gentle movement + mobility work (10 min)",
                "Main: Low-moderate intensity at 60-70% effort",
                "Focus: Form correction and base conditioning",
                "Cooldown: Extended stretching + relaxation (20 min)"
            ])
        else:
            plan.extend([
                "Active recovery: Light walking or gentle yoga (20 min)",
                "Focus: Breathing exercises and mental relaxation",
                "Hydration: Increase fluid intake throughout day",
                "Rest: Prioritize 8+ hours of quality sleep"
            ])
        
        # Add sport-specific modifications
        if profile and profile.sport_type:
            sport_specific = get_sport_specific_recommendations(profile.sport_type, score)
            plan.extend(sport_specific)
        
        return plan[:6]  # Limit to 6 recommendations
        
    except Exception as e:
        return [
            "Warm-up: 10 minutes dynamic preparation",
            "Main: Train at appropriate intensity for your current state",
            "Cooldown: 15 minutes recovery and stretching"
        ]

def get_sport_specific_recommendations(sport_type: SportType, score: float) -> List[str]:
    """Get sport-specific training recommendations"""
    recommendations = {
        SportType.ENDURANCE: [
            "Zone 2 cardio focus for aerobic base building",
            "Include interval training if score > 70"
        ],
        SportType.STRENGTH: [
            "Compound movements with proper form emphasis",
            "Reduce load by 10-20% if fatigue is high"
        ],
        SportType.TEAM_SPORTS: [
            "Agility drills and sport-specific movements",
            "Team coordination exercises if energy permits"
        ],
        SportType.INDIVIDUAL: [
            "Technical skill refinement sessions",
            "Mental visualization and focus training"
        ],
        SportType.COMBAT: [
            "Pad work and technique drilling",
            "Conditioning work based on readiness level"
        ],
        SportType.MIXED: [
            "Varied training combining strength and cardio",
            "Functional movement patterns"
        ]
    }
    
    return recommendations.get(sport_type, ["Sport-specific skill practice"])

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Additional utility endpoints
@app.route('/voices/test', methods=['POST'])
def test_voice():
    """Test voice generation"""
    try:
        data = request.json
        text = data.get('text', 'This is a test message from Coach Alex.')
        
        audio_file = coach_service.generate_voice_audio(text, "test")
        
        if audio_file:
            return jsonify({
                "success": True,
                "audio_url": f'/media/{os.path.basename(audio_file)}'
            })
        else:
            return jsonify({
                "success": False,
                "message": "Voice generation not available"
            })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Get system configuration"""
    try:
        config_info = {
            "model_name": Config.MODEL_NAME,
            "api_base": Config.API_BASE,
            "elevenlabs_available": bool(Config.ELEVENLABS_API_KEY and Config.ELEVENLABS_API_KEY != "your_elevenlabs_api_key_here"),
            "voice_id": Config.ELEVENLABS_VOICE_ID,
            "max_response_length": Config.MAX_RESPONSE_LENGTH,
            "session_duration": Config.SESSION_DURATION_MINUTES
        }
        
        return jsonify(config_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up old audio files"""
    try:
        cleanup_count = 0
        current_time = time.time()
        
        # Clean files older than 1 hour
        max_age = 3600
        
        if os.path.exists(Config.UPLOAD_FOLDER):
            for filename in os.listdir(Config.UPLOAD_FOLDER):
                file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age:
                        os.remove(file_path)
                        cleanup_count += 1
        
        return jsonify({
            "message": f"Cleaned up {cleanup_count} old files",
            "count": cleanup_count
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Development mode features
if __name__ == '__main__':
    import sys
    
    # Check if Ollama is running
    try:
        response = requests.get(f"{Config.API_BASE}/api/tags", timeout=5)
        if response.status_code != 200:
            print("⚠️  Warning: Ollama server not responding. Make sure it's running:")
            print("   ollama serve")
            print("   ollama pull llama3.2:latest")
    except requests.exceptions.RequestException:
        print("❌ Error: Cannot connect to Ollama server.")
        print("   Please start Ollama first:")
        print("   ollama serve")
        sys.exit(1)
    
    # Check ElevenLabs configuration
    if Config.ELEVENLABS_API_KEY == "your_elevenlabs_api_key_here":
        print("⚠️  Warning: ElevenLabs API key not configured.")
        print("   Set ELEVENLABS_API_KEY environment variable for voice features.")
    else:
        print("✅ ElevenLabs configuration found.")
    
    print("🎤 Starting ActiveMind Sport Voice Coach Backend...")
    print(f"   Model: {Config.MODEL_NAME}")
    print(f"   Database: {Config.DATABASE_PATH}")
    print(f"   Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"   Voice ID: {Config.ELEVENLABS_VOICE_ID}")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True,
        threaded=True
    )