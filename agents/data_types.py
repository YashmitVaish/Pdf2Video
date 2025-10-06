from dataclasses import dataclass
from typing import List, Optional , Dict , Any

@dataclass
class Page:
    page_number: int
    section_title: str
    text: str
    tables: List[Dict[str, Any]]

@dataclass
class Narration:
    page_number: int
    text: str
    audio_path: Optional[str] = None
    duration: float = 0.0
