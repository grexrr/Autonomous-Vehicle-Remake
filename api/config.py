import os
from typing import List
from dotenv import load_dotenv 

load_dotenv()
# =============== Basic Config =============== 
class Config:
    # Flask App key (for session encryption, etc) 
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # CORS authorized domains
    CORS_ORIGINS: List[str] = []

    # WebSocket Config
    SOCKETIO_CORS_ALLOWED_ORIGINS: List[str] = []


class DevelopmentConfig(Config):
    DEBUG = True
    PORT = int(os.getenv('PORT', 5000))

    CORS_ORIGINS = [
        'http://localhost:5173',
        'http://localhost:8080',  # For testing
    ]

    SOCKETIO_CORS_ALLOWED_ORIGINS = CORS_ORIGINS

class ProductionConfig(Config):
    DEBUG = False 
    
    CORS_ORIGINS = os.getenv('ALLOWED_ORIGINS', '').split(',')
    SOCKETIO_CORS_ALLOWED_ORIGINS = CORS_ORIGINS

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig  
}