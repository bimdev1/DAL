"""SDT (Segment Descriptor Tag) Injection for DAL.

This module provides functionality to build prompts based on SDT attributes
and apply them to text generation requests.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

from .dal_types import DALTagBlock

@dataclass
class SDTApplied:
    """Tracks which SDT attributes were applied during prompt construction."""
    tone: bool = False
    depth: bool = False
    format: bool = False
    style: bool = False
    custom_instructions: bool = False
    
    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary for serialization."""
        return asdict(self)

def build_prompt(text: str, sdt: DALTagBlock) -> Tuple[str, SDTApplied]:
    """Build a prompt from input text and SDT attributes.
    
    Args:
        text: The input text to process
        sdt: DALTagBlock containing SDT attributes
        
    Returns:
        Tuple of (prompt_text, sdt_applied) where:
        - prompt_text: The constructed prompt with SDT instructions
        - sdt_applied: SDTApplied object showing which attributes were used
    """
    if not sdt or not hasattr(sdt, 'tags'):
        return text, SDTApplied()
    
    sdt_applied = SDTApplied()
    instructions = []
    
    # Check for tone specification
    if hasattr(sdt, 'tone') and sdt.tone and sdt.tone != "default":
        instructions.append(f"Use a {sdt.tone} tone")
        sdt_applied.tone = True
    
    # Check for depth specification
    if hasattr(sdt, 'depth') and sdt.depth and sdt.depth != "default":
        depth_map = {
            "brief": "Be concise and to the point",
            "normal": "Provide a balanced level of detail",
            "detailed": "Be thorough and comprehensive",
            "exhaustive": "Provide extremely detailed and exhaustive coverage"
        }
        if sdt.depth in depth_map:
            instructions.append(depth_map[sdt.depth])
            sdt_applied.depth = True
    
    # Check for format specification
    if hasattr(sdt, 'format') and sdt.format and sdt.format != "default":
        format_map = {
            "paragraph": "Format as a well-structured paragraph",
            "bullets": "Format as bullet points",
            "steps": "Format as numbered steps",
            "table": "Format as a table if appropriate",
            "code": "Format as code with appropriate syntax highlighting"
        }
        if sdt.format in format_map:
            instructions.append(format_map[sdt.format])
            sdt_applied.format = True
    
    # Check for style specification
    if hasattr(sdt, 'style') and sdt.style and sdt.style != "default":
        instructions.append(f"Write in a {sdt.style} style")
        sdt_applied.style = True
    
    # Check for custom instructions
    if hasattr(sdt, 'instructions') and sdt.instructions:
        instructions.append(sdt.instructions)
        sdt_applied.custom_instructions = True
    
    # If no SDT attributes were applied, return original text
    if not any(asdict(sdt_applied).values()):
        return text, sdt_applied
    
    # Build the prompt
    instruction_str = ". ".join(instructions) + "."
    
    prompt = f"""[INST] 
You are a helpful AI assistant. Rewrite the following text with these instructions:

{instruction_str}

Original text:
{text}

Rewritten text: [/INST]"""
    
    return prompt, sdt_applied
