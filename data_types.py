from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Page:
    number: int
    image_path: str
    text: str

@dataclass
class Narration:
    page_number: int
    text: str
    audio_path: Optional[str] = None
    duration: float = 0.0
