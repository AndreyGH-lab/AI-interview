import mimetypes

import requests

from shared.config import STT_URL

def transcribe_audio(path: str, language: str = "ru") -> str:
    """
    Отправляет аудиофайл в STT-сервис и возвращает распознанный текст.
    """
    mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    with open(path, "rb") as f:
        files = {"file": (path, f, mime_type)}
        data = {"language": language}
        resp = requests.post(STT_URL, files=files, data=data, timeout=300)
        resp.raise_for_status()
        j = resp.json()
        return j["text"]
