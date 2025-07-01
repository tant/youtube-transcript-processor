"""Helper functions for processing YouTube transcripts.

This module provides utilities for:
- Transcript selection and extraction
- Text cleaning and formatting
- Gemini AI integration
- Error handling and logging
"""

from typing import Tuple, List, Dict, Optional, Any
import re
import logging
from fastapi import HTTPException
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    VideoUnavailable
)
import google.generativeai as genai

# Configure module logging
logger = logging.getLogger(__name__)

# Global Gemini model instance
_model = None

def configure_gemini(api_key: str) -> Optional[Any]:
    """Configure Gemini AI with API key and return model.
    
    Args:
        api_key: Gemini API key
        
    Returns:
        Optional[Any]: Configured model or None if failed
    """
    try:
        global _model
        genai.configure(api_key=api_key)
        
        # List available models
        model_list = genai.list_models()
        logger.info("Available Gemini Models: %s", 
                   [model.name for model in model_list])
        
        # Initialize the model
        _model = genai.GenerativeModel('gemini-pro')
        logger.info("Successfully initialized Gemini model")
        return _model
    except Exception as e:
        logger.error("Failed to configure Gemini: %s", str(e))
        return None

def get_model() -> Optional[Any]:
    """Get the configured Gemini model instance."""
    global _model
    return _model

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats.
    
    Args:
        url (str): The YouTube URL to extract ID from
        
    Returns:
        str: The extracted video ID, or None if invalid URL
    """
    if not url:
        return None
        
    # Check if it's a YouTube URL
    if not any(domain in url.lower() for domain in ['youtube.com', 'youtu.be']) and len(url) != 11:
        return None
        
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&?\s]|$)',  # Regular URL or embed URL
        r'^([0-9A-Za-z_-]{11})$'  # Just the video ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match and len(match.group(1)) == 11:  # Ensure exact length
            return match.group(1)
    return None

def generate_short_summary(text: str) -> str:
    """Generate a short 100-word summary of the transcript using Gemini.
    
    Args:
        text (str): The full transcript text to summarize
        
    Returns:
        str: A 100-word summary of the content
        
    Raises:
        ValueError: If Gemini model is not configured or generation fails
    """
    model = get_model()
    if not model:
        raise ValueError("Gemini model not properly configured")
        
    prompt = """Create a clear and concise 100-word summary of this transcript. Focus on:
- Main topics and key points
- Important conclusions or insights
- Keep it factual and objective
- Exactly 100 words

Text to summarize:
{}""".format(text)

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise ValueError(f"Failed to generate summary: {str(e)}")

def get_available_transcripts(video_id: str) -> List[Dict]:
    """Get list of available transcripts for a video.
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        list: List of available transcript languages and types
        
    Note:
        Returns empty list if video is unavailable or has no transcripts
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available = []
        
        # Add all available transcripts
        for transcript in transcript_list:
            available.append({
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "is_translatable": transcript.is_translatable
            })
            
        return available
        
    except VideoUnavailable:
        logger.warning("Video %s is unavailable", video_id)
        return []
    except TranscriptsDisabled:
        logger.warning("Transcripts are disabled for video %s", video_id)
        return []
    except Exception as e:
        logger.error("Error fetching transcripts for video %s: %s", video_id, str(e))
        return []

def get_best_transcript(video_id: str) -> Tuple[List, Dict]:
    """Get the best available transcript for a video.
    
    Priority order:
    1. Any manual transcript (preferred)
    2. Any auto-generated transcript
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        tuple: (transcript_dict, transcript_info)
            transcript_dict: The transcript content
            transcript_info: Dict with language and type info
        
    Raises:
        HTTPException: If no suitable transcript is found
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_transcripts = []
        first_transcript = None
        
        logger.debug("Scanning available transcripts for video %s", video_id)
        
        # Get list of all transcripts 
        for transcript in transcript_list:
            logger.debug("Found transcript: Language=%s (%s), Generated=%s, Translatable=%s",
                       transcript.language,
                       transcript.language_code,
                       transcript.is_generated,
                       transcript.is_translatable)
            
            # Store first transcript as fallback
            if first_transcript is None:
                first_transcript = transcript
                logger.debug("Stored first transcript as fallback: %s (%s)",
                           transcript.language,
                           transcript.language_code)
            
            if not transcript.is_generated:
                # Found a manual transcript, try to use it
                logger.info("Found manual transcript for %s: %s (%s)",
                          video_id,
                          transcript.language,
                          transcript.language_code)
                try:
                    return (
                        transcript.fetch(),
                        {
                            "language": transcript.language,
                            "language_code": transcript.language_code,
                            "type": "manual",
                            "is_generated": False
                        }
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to fetch manual transcript for %s (%s): %s",
                        video_id,
                        transcript.language_code,
                        str(e)
                    )
                    # Continue looking for other transcripts
            
            available_transcripts.append({
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "is_translatable": transcript.is_translatable
            })
            
        # If we get here, no manual transcript was found or failed to fetch
        # Use first available transcript if we have one
        if first_transcript:
            logger.info(
                "Using first available transcript for %s: %s (%s)", 
                video_id,
                first_transcript.language,
                first_transcript.language_code
            )
            try:
                return (
                    first_transcript.fetch(),
                    {
                        "language": first_transcript.language,
                        "language_code": first_transcript.language_code,
                        "type": "auto",
                        "is_generated": True
                    }
                )
            except Exception as e:
                logger.error(
                    "Failed to fetch auto-generated transcript for %s (%s): %s",
                    video_id,
                    first_transcript.language_code,
                    str(e)
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to fetch transcript: {str(e)}"
                )
            
    except TranscriptsDisabled:
        logger.error("Transcripts are disabled for video %s", video_id)
        raise HTTPException(
            status_code=404,
            detail="Transcripts are disabled for this video"
        )
    except VideoUnavailable:
        logger.error("Video %s is unavailable", video_id)
        raise HTTPException(
            status_code=404,
            detail="Video is unavailable"
        )
    except Exception as e:
        logger.error("Unexpected error getting transcript for %s: %s",
                    video_id, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching transcript: {str(e)}"
        )

def clean_transcript_text(transcript_text: List[str]) -> List[str]:
    """Clean transcript text by removing sound descriptions and formatting.
    
    Args:
        transcript_text (list): List of transcript segments
        
    Returns:
        list: Cleaned transcript segments
        
    Raises:
        ValueError: If input is not a list or contains non-string elements
    """
    # Type validation
    if not isinstance(transcript_text, list):
        raise ValueError(f"Expected list input, got {type(transcript_text)}")
    
    cleaned_text = []
    sound_patterns = [r'\[Music\]', r'\[Applause\]', r'\[.*?\]']
    
    for text in transcript_text:
        try:
            # Convert to string if possible
            text_str = str(text) if text is not None else ""
            
            # Remove sound descriptions
            clean_text = text_str
            for pattern in sound_patterns:
                clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
            
            # Remove newlines and extra spaces
            clean_text = clean_text.replace('\n', ' ').strip()
            
            # Add if not empty
            if clean_text:
                cleaned_text.append(clean_text)
                
        except Exception as e:
            logger.warning("Failed to process transcript segment: %s", str(e))
            continue
            
    return cleaned_text

def extract_transcript_segments(transcript_data) -> List[str]:
    """Extract text from transcript data.
    
    Args:
        transcript_data: Can be one of:
            - List[FetchedTranscriptSnippet]
            - List[dict]
            - FetchedTranscript (direct output from fetch())
        
    Returns:
        list: List of text segments
        
    Raises:
        HTTPException: If text extraction fails
        ValueError: If input contains invalid segments
    """
    logger.debug("Processing transcript data of type: %s", type(transcript_data))
    
    # Handle FetchedTranscript object from youtube_transcript_api
    if str(type(transcript_data)) == "<class 'youtube_transcript_api._transcripts.FetchedTranscript'>":
        logger.debug("Input is FetchedTranscript from youtube_transcript_api")
        try:
            logger.debug("Available attributes: %s", dir(transcript_data))
            
            # Try to access the transcript segments directly
            segments = []
            
            # Try to access raw property if available
            if hasattr(transcript_data, '_transcripts') and isinstance(transcript_data._transcripts, list):
                logger.debug("Found _transcripts as list")
                segments = transcript_data._transcripts
            elif hasattr(transcript_data, 'snippets'):
                logger.debug("Found snippets attribute")
                segments = transcript_data.snippets
            elif str(transcript_data).startswith('FetchedTranscript(snippets=['):
                logger.debug("Parsing str representation")
                # Parse the string representation which contains all segments
                # Format is: FetchedTranscript(snippets=[FetchedTranscriptSnippet(text='...', start=X, duration=Y), ...])
                str_repr = str(transcript_data)
                segments_str = str_repr[str_repr.find('[') + 1:str_repr.rfind(']')]
                # Split by ), but not if inside quotes
                parts = []
                current = ""
                in_quotes = False
                for char in segments_str:
                    if char == "'" and segments_str[segments_str.find(char)-1] != '\\':
                        in_quotes = not in_quotes
                    if char == ')' and not in_quotes and current.strip():
                        parts.append(current + ')')
                        current = ""
                        continue
                    current += char
                if current.strip():
                    parts.append(current)
                    
                # Process each part to extract text
                for part in parts:
                    if 'text=' in part:
                        # Extract text between quotes after text=
                        text_start = part.find("text='") + 6
                        text_end = part.find("'", text_start)
                        text = part[text_start:text_end]
                        if text:
                            segments.append({'text': text})
            
            if segments:
                logger.debug("Found %d segments", len(segments))
                transcript_data = segments
            else:
                logger.warning("Could not extract segments, using empty list")
                transcript_data = []
        except Exception as e:
            logger.error("Failed to extract from FetchedTranscript: %s", str(e))
            logger.debug("Object details: %s", dir(transcript_data))
            raise ValueError(f"Failed to process FetchedTranscript: {str(e)}")
    
    # Convert single dict/object to list
    if isinstance(transcript_data, dict) or hasattr(transcript_data, 'text'):
        logger.debug("Converting single segment to list")
        transcript_data = [transcript_data]
    
    # Ensure we have a list
    if not isinstance(transcript_data, list):
        logger.error("Expected list or FetchedTranscript, got %s", type(transcript_data))
        raise ValueError(f"Expected list or FetchedTranscript input, got {type(transcript_data)}")
    
    transcript_text = []
    try:
        logger.debug("Processing %d segments", len(transcript_data))
        for segment in transcript_data:
            text = None
            logger.debug("Processing segment of type: %s", type(segment))
            logger.debug("Segment content: %s", segment)
            
            # First try the object interface (FetchedTranscriptSnippet)
            if hasattr(segment, 'text'):
                try:
                    text = getattr(segment, 'text')
                    logger.debug("Extracted text using attribute: %s", text[:50])
                except AttributeError as ae:
                    logger.warning("Failed to access text attribute: %s", str(ae))
                    logger.debug("Available attributes: %s", dir(segment))
                    continue
            
            # Try dict interface with fallbacks
            elif isinstance(segment, dict):
                # Try common key variations
                keys_to_try = ['text', 'content', 'transcript', 'value']
                for key in keys_to_try:
                    text = segment.get(key)
                    if text is not None:
                        logger.debug("Found text using key '%s': %s", key, text[:50])
                        break
                if text is None:
                    logger.warning("Could not find text in dict with keys: %s", list(segment.keys()))
                    continue
                    
            else:
                try:
                    # Last resort - try string conversion
                    text = str(segment).strip()
                    if text and not text.startswith('<') and not text.endswith('>'):
                        logger.debug("Converted segment to string: %s", text[:50])
                    else:
                        logger.warning("Segment conversion yielded invalid text")
                        continue
                except Exception as e:
                    logger.warning("Could not process segment of type %s: %s", type(segment), str(e))
                    continue
                
            # Ensure we have text
            if text is None:
                continue
                
            # Convert to string safely
            try:
                text_str = str(text).strip()
                if text_str:  # Only append non-empty strings
                    transcript_text.append(text_str)
                    logger.debug("Added text segment: %s", text_str[:50])
            except Exception as e:
                logger.warning("Failed to convert text to string: %s", str(e))
                logger.debug("Problematic text content: %s", text)
                continue
                
        if not transcript_text:
            raise ValueError("No valid text segments found in transcript")
            
        return transcript_text
                
    except ValueError as ve:
        # Re-raise ValueError as HTTPException
        raise HTTPException(
            status_code=500,
            detail=str(ve)
        )
    except Exception as e:
        logger.error("Error extracting text from transcript: %s", str(e))
        logger.debug("Full error context: %s: %s", str(e.__class__.__name__), str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from transcript: {str(e)}"
        )

def format_transcript_with_gemini(text: str) -> str:
    """Format transcript text into proper paragraphs using Gemini.
    
    Args:
        text (str): Raw transcript text to format
        
    Returns:
        str: Formatted text with proper paragraphs in Markdown
        
    Raises:
        ValueError: If Gemini model is not configured or generation fails
    """
    model = get_model()
    if not model:
        raise ValueError("Gemini model not properly configured")

    prompt = """Reconstruct these transcript segments into a well-formatted text. Rules:
- Keep 100% of original words
- Only fix: connections between segments, spacing, and punctuation
- No summarizing, no new content
- Group related content into logical paragraphs
- Format output as Markdown with proper paragraphs
- Use two newlines between paragraphs for proper Markdown formatting
- For any speaker changes or major topic shifts, start a new paragraph
- Preserve any quotes or important phrases exactly as they appear
- Keep all Unicode characters and special symbols unchanged

Text to reconstruct:
{}""".format(text)

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Validate output length
        if len(result) < len(text) * 0.9:
            logger.warning("Generated text appears to be missing content")
            return text  # Return original if too much content was lost
            
        return result
    except Exception as e:
        logger.error("Error formatting transcript: %s", str(e))
        return text  # Return original text as fallback
