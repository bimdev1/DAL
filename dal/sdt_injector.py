"""SDT (Segment Descriptor Tag) Injection for DAL.

This module provides functionality to build prompts based on SDT attributes
and apply them to text generation requests with proper normalization and formatting.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Tuple, List, Literal
import logging

from .dal_types import DALTagBlock

logger = logging.getLogger(__name__)

# Type aliases
PromptStyle = Literal['llama_inst', 'chatml', 'plain']
Tone = Literal['neutral', 'enthusiastic', 'professional', 'skeptical']
Format = Literal['exposition', 'steps', 'bullet_list', 'tabular']
Depth = Literal[1, 3, 5]

# Default constraints
DEFAULT_CONSTRAINTS = [
    "Be factually accurate and avoid hallucinations",
    "State uncertainties or knowledge gaps clearly",
    "Maintain consistency with previous context",
    "Be concise and avoid unnecessary repetition"
]

@dataclass
class SDTOpts:
    """Options for SDT prompt construction."""
    prompt_style: PromptStyle = 'llama_inst'
    include_constraints: bool = True
    extra_constraints: List[str] = field(default_factory=list)
    backend_hint: Optional[str] = None  # e.g., 'huggingface', 'openai'

@dataclass
class SDTApplied:
    """Tracks which SDT attributes were applied during prompt construction."""
    tone: Optional[Tone] = None
    depth: Optional[Depth] = None
    format: Optional[Format] = None
    style: Optional[str] = None
    custom_instructions: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'tone': self.tone,
            'depth': self.depth,
            'format': self.format,
            'style': self.style,
            'custom_instructions': self.custom_instructions,
            'constraints': self.constraints
        }

def _normalize_tone(tone: Optional[str]) -> Optional[Tone]:
    """Normalize tone to one of the allowed values."""
    if not tone or tone == 'default':
        return None
    
    tone = tone.lower()
    if tone in ['neutral', 'enthusiastic', 'professional', 'skeptical']:
        return tone  # type: ignore
    
    # Map common alternatives
    tone_map = {
        'formal': 'professional',
        'casual': 'neutral',
        'excited': 'enthusiastic',
        'doubtful': 'skeptical',
        'skeptical': 'skeptical',
    }
    
    return tone_map.get(tone, 'neutral')  # type: ignore

def _normalize_format(fmt: Optional[str]) -> Optional[Format]:
    """Normalize format to one of the allowed values."""
    if not fmt or fmt == 'default':
        return None
    
    fmt = fmt.lower()
    if fmt in ['exposition', 'steps', 'bullet_list', 'tabular']:
        return fmt  # type: ignore
    
    # Map common alternatives
    fmt_map = {
        'paragraph': 'exposition',
        'bullet': 'bullet_list',
        'bullets': 'bullet_list',
        'number': 'steps',
        'numbered': 'steps',
        'table': 'tabular',
        'code': 'exposition'  # Special case handled separately
    }
    
    return fmt_map.get(fmt, 'exposition')  # type: ignore

def _normalize_depth(depth: Any) -> Optional[Depth]:
    """Normalize depth to 1, 3, or 5."""
    if not depth or depth == 'default':
        return None
    
    try:
        depth_int = int(depth)
        if depth_int <= 2:
            return 1
        elif depth_int <= 4:
            return 3
        return 5
    except (ValueError, TypeError):
        depth_str = str(depth).lower()
        if 'brief' in depth_str or 'summary' in depth_str:
            return 1
        if 'detailed' in depth_str or 'thorough' in depth_str:
            return 5
        return 3  # Default to medium depth

def _format_llama_inst(instructions: List[str], text: str) -> str:
    """Format prompt in LLaMA instruction style."""
    instruction_text = "\n".join(instructions)
    return f"""<|system|>
You are a helpful AI assistant. Rewrite the following text according to these instructions:
{instruction_text}

IMPORTANT: Only respond with the rewritten text. DO NOT repeat or include the original text in your response.
DO NOT include any additional commentary, explanations, or formatting.

Original text to rewrite:
{text}
<|assistant|>
"""

def _format_chatml(instructions: List[str], text: str) -> str:
    """Format prompt in ChatML style."""
    instruction_text = "\n".join(instructions)
    return f"""<|im_start|>system
You are a helpful AI assistant. Rewrite the following text according to these instructions:
{instruction_text}

IMPORTANT: Only respond with the rewritten text. DO NOT repeat or include the original text in your response.
DO NOT include any additional commentary, explanations, or formatting.<|im_end|>
<|im_start|>user
Original text to rewrite:
{text}<|im_end|>
<|im_start|>assistant
"""

def _format_plain(instructions: List[str], text: str) -> str:
    """Format prompt in plain text style."""
    instruction_text = "\n".join(instructions)
    return f"""You are a helpful AI assistant. Rewrite the following text according to these instructions:
{instruction_text}

IMPORTANT: Only respond with the rewritten text. DO NOT repeat or include the original text in your response.
DO NOT include any additional commentary, explanations, or formatting.

Original text to rewrite:
{text}

Rewritten text (ONLY the rewritten text, no quotes or formatting):
"""

def build_prompt(
    text: str, 
    sdt: DALTagBlock, 
    opts: Optional[SDTOpts] = None
) -> Tuple[str, SDTApplied]:
    """Build a prompt from input text and SDT attributes.
    
    Args:
        text: The input text to process
        sdt: DALTagBlock containing SDT attributes
        opts: Optional SDTOpts for prompt construction
        
    Returns:
        Tuple of (prompt_text, sdt_applied) where:
        - prompt_text: The constructed prompt with SDT instructions
        - sdt_applied: SDTApplied object showing which attributes were used
    """
    if not opts:
        opts = SDTOpts()
    
    # Check if we have any SDT attributes to process
    has_attributes = False
    if sdt and hasattr(sdt, 'attributes'):
        has_attributes = any(
            hasattr(sdt.attributes, attr) 
            for attr in ['tone', 'depth', 'format', 'style', 'instructions']
        )
    
    if not has_attributes:
        return text, SDTApplied()
    
    sdt_applied = SDTApplied()
    instructions = []
    
    # Process tone with more specific instructions
    if hasattr(sdt.attributes, 'tone') and sdt.attributes.tone:
        normalized_tone = _normalize_tone(sdt.attributes.tone)
        if normalized_tone:
            tone_instructions = {
                'neutral': 'Maintain a neutral, objective tone throughout. Avoid emotional language and focus on facts.',
                'enthusiastic': 'Use an enthusiastic and engaging tone. Show excitement and interest in the topic.',
                'professional': 'Adopt a formal, professional tone suitable for business or academic contexts.',
                'skeptical': 'Adopt a questioning, critical perspective. Examine claims carefully and point out potential issues.',
            }
            instructions.append(tone_instructions.get(normalized_tone, ''))
            sdt_applied.tone = normalized_tone
    
    # Process depth with more specific guidance
    if hasattr(sdt.attributes, 'depth') and sdt.attributes.depth is not None:
        normalized_depth = _normalize_depth(sdt.attributes.depth)
        if normalized_depth:
            depth_instructions = {
                1: 'Be extremely concise (1-2 sentences max). Focus only on the most critical information.',
                3: 'Provide a balanced level of detail (3-5 key points). Include essential context but stay focused.',
                5: 'Be thorough and comprehensive. Cover all important aspects, include examples, and address potential counterpoints.',
            }
            instructions.append(depth_instructions[normalized_depth])
            sdt_applied.depth = normalized_depth
    
    # Process format with clearer instructions
    if hasattr(sdt.attributes, 'format') and sdt.attributes.format:
        normalized_format = _normalize_format(sdt.attributes.format)
        if normalized_format:
            format_instructions = {
                'paragraph': 'Format as a well-structured, coherent paragraph with smooth transitions between ideas.',
                'bullets': 'Format as a clear bulleted list. Each bullet should be concise and start with a key point.',
                'steps': 'Format as a numbered list of sequential steps. Each step should be a complete action.',
                'table': 'Format as a structured table with clear headers. Each row should represent a distinct item or concept, and columns should organize related information.',
            }
            if normalized_format in format_instructions:
                instructions.append(format_instructions[normalized_format])
                sdt_applied.format = normalized_format
    
    # Process style with more specific guidance
    if hasattr(sdt.attributes, 'style') and sdt.attributes.style:
        style = sdt.attributes.style.lower()
        style_instructions = {
            'academic': 'Use formal academic language with appropriate terminology. Support claims with evidence and cite sources where applicable.',
            'casual': 'Use a conversational, friendly tone as if explaining to a colleague. Contractions and first-person are acceptable.',
            'technical': 'Use precise technical terminology. Assume the reader has domain knowledge. Include relevant specifications or technical details.',
            'narrative': 'Use a storytelling approach. Set the scene, introduce characters or concepts, and build toward a clear point or conclusion.',
        }
        if style in style_instructions:
            instructions.append(style_instructions[style])
            sdt_applied.style = style
    
    # Process custom instructions with better handling
    if hasattr(sdt.attributes, 'instructions') and sdt.attributes.instructions:
        if isinstance(sdt.attributes.instructions, str):
            if sdt.attributes.instructions.strip():
                instructions.append(sdt.attributes.instructions.strip())
                sdt_applied.custom_instructions = sdt.attributes.instructions.strip()
        elif isinstance(sdt.attributes.instructions, list):
            valid_instructions = [i.strip() for i in sdt.attributes.instructions if i and i.strip()]
            if valid_instructions:
                instructions.extend(valid_instructions)
                sdt_applied.custom_instructions = '\n'.join(valid_instructions)
    
    # Add any extra constraints from options
    if opts.include_constraints:
        constraints = [
            'Be accurate and factual',
            'Be clear and concise',
            'Maintain a professional tone',
        ]
        if opts.extra_constraints:
            constraints.extend(opts.extra_constraints)
        instructions.extend(constraints)
        sdt_applied.constraints = constraints
    
    # If no instructions were added, return the original text
    if not instructions:
        return text, sdt_applied
    
    # Format the final prompt based on style
    formatters = {
        'llama_inst': _format_llama_inst,
        'chatml': _format_chatml,
        'plain': _format_plain,
    }
    
    formatter = formatters.get(opts.prompt_style, _format_plain)
    return formatter(instructions, text), sdt_applied
