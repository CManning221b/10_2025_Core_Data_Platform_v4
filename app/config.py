import os
from datetime import timedelta


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', '121963')
    DEBUG = False
    UPLOAD_FOLDER = os.path.abspath('./uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    UPLOAD_FOLDER = os.path.abspath('./test_uploads')


config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}