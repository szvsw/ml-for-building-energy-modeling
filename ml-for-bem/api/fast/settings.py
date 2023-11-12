from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """General API Settings."""

    api_key: str = Field(..., validation_alias="API_KEY")
