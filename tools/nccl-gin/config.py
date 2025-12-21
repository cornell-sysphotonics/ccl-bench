# torchtitan/components/gin/config.py
from dataclasses import dataclass

@dataclass
class GINConfig:
    enabled: bool = False
