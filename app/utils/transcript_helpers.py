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
import requests

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
        
        # Initialize the model
        _model = genai.GenerativeModel('gemini-2.5-flash')
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
        
        for transcript in transcript_list:
            if first_transcript is None:
                first_transcript = transcript
            
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

def clean_transcript_text(transcript_text: list) -> str:
    """Clean and reconstruct transcript text using Gemini AI.

    Args:
        transcript_text (list): A list of transcript text segments (strings).

    Returns:
        str: A single string with the cleaned and formatted transcript.
    """
    # First, join all text segments into a single block
    full_text = " ".join(transcript_text)

    # Use Gemini to clean and reformat the text
    model = get_model()
    if not model:
        raise ValueError("Gemini model not properly configured")

    prompt = f"""Please clean and reformat the following transcript. The transcript is currently a single block of text with potential errors, missing punctuation, and no paragraph breaks. Your task is to:

1.  **Correct grammar and spelling mistakes.**
2.  **Add appropriate punctuation**, including commas, periods, and question marks.
3.  **Structure the text into logical paragraphs.** Each paragraph should represent a coherent idea or a speaker's turn.
4.  **Do not summarize or change the meaning.** Preserve the original language and content.
5.  **Ensure the output is a single, clean block of text.**

Here is the transcript:

{full_text}
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Failed to clean transcript with Gemini: {str(e)}")
        # Fallback to simple join if AI fails
        return full_text

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
    
    if str(type(transcript_data)) == "<class 'youtube_transcript_api._transcripts.FetchedTranscript'>":
        try:
            segments = []
            
            if hasattr(transcript_data, '_transcripts') and isinstance(transcript_data._transcripts, list):
                segments = transcript_data._transcripts
            elif hasattr(transcript_data, 'snippets'):
                segments = transcript_data.snippets
            elif str(transcript_data).startswith('FetchedTranscript(snippets=['):
                str_repr = str(transcript_data)
                segments_str = str_repr[str_repr.find('[') + 1:str_repr.rfind(']')]
                
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
                    
                for part in parts:
                    if 'text=' in part:
                        text_start = part.find("text='") + 6
                        text_end = part.find("'", text_start)
                        text = part[text_start:text_end]
                        if text:
                            segments.append({'text': text})
            
            if segments:
                transcript_data = segments
            else:
                transcript_data = []
        except Exception as e:
            logger.error("Failed to extract from FetchedTranscript: %s", str(e))
            raise ValueError(f"Failed to process FetchedTranscript: {str(e)}")
    
    if isinstance(transcript_data, dict) or hasattr(transcript_data, 'text'):
        transcript_data = [transcript_data]
    
    if not isinstance(transcript_data, list):
        logger.error("Expected list or FetchedTranscript, got %s", type(transcript_data))
        raise ValueError(f"Expected list or FetchedTranscript input, got {type(transcript_data)}")
    
    transcript_text = []
    try:
        for segment in transcript_data:
            text = None
            
            if hasattr(segment, 'text'):
                try:
                    text = getattr(segment, 'text')
                except AttributeError as ae:
                    logger.warning("Failed to access text attribute: %s", str(ae))
                    continue
            
            elif isinstance(segment, dict):
                keys_to_try = ['text', 'content', 'transcript', 'value']
                for key in keys_to_try:
                    text = segment.get(key)
                    if text is not None:
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

def get_video_info(video_id: str) -> Dict:
    """Get basic video information from YouTube using oEmbed API.
    
    Returns dict with title, author name, author URL, and thumbnail URL.
    Returns empty values if video info cannot be retrieved."""
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url)
        response.raise_for_status()
        data = response.json()
        return {
            "title": data.get("title"),
            "author_name": data.get("author_name"),
            "author_url": data.get("author_url"),
            "thumbnail_url": data.get("thumbnail_url")
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch video info for {video_id}: {e}")
        return {
            "title": None,
            "author": None,
            "published_at": None
        }
    except Exception as e:
        logger.error("Error getting video info: %s", str(e))
        return {
            "title": None,
            "author": None,
            "published_at": None
        }
