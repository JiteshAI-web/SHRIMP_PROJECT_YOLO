"""
Database Configuration for Shrimp Detection System
PostgreSQL Database
"""

import os
from dataclasses import dataclass
from urllib.parse import quote_plus


@dataclass
class DatabaseConfig:
    """Database configuration class"""
    username: str = os.getenv("DB_USER", "bariflo")
    password: str = os.getenv("DB_PASSWORD", "Bariflo@2401")
    host: str = os.getenv("DB_HOST", "192.168.1.199")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "shrimp_db")
    
    def _get_encoded_password(self) -> str:
        """URL-encode the password to handle special characters"""
        return quote_plus(self.password)
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        encoded_password = self._get_encoded_password()
        return f"postgresql://{self.username}:{encoded_password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        """Generate async PostgreSQL connection string for asyncpg"""
        encoded_password = self._get_encoded_password()
        return f"postgresql+asyncpg://{self.username}:{encoded_password}@{self.host}:{self.port}/{self.database}"


# Global config instance
db_config = DatabaseConfig()
