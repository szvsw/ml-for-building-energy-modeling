from api.fast import settings
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="Authorization", auto_error=True)


async def get_api_key(api_key: str = Security(api_key_header)):
    """Compare provided key against api key."""
    if api_key == settings.api_key:
        return api_key
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
