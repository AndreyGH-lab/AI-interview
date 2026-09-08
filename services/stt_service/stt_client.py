import requests

from shared.config import STT_URL

def transcribe_audio(path: str, language: str = "ru") -> str:
    """
    Отправляет аудиофайл в STT-сервис и возвращает распознанный текст.
    """
    with open(path, "rb") as f:
        files = {"file": (path, f, "audio/wav")}
        data = {"language": language}
        resp = requests.post(STT_URL, files=files, data=data, timeout=300)
        resp.raise_for_status()
        j = resp.json()
        return j["text"]
