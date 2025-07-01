from typing import Tuple, List, Dict, Optional, Any
import re
from fastapi import HTTPException
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    VideoUnavailable
)
import google.generativeai as genai

# Initialize model variable
_model: Optional[Any] = None

def get_model() -> Optional[Any]:
    """Get the configured Gemini model instance.
    
    Returns:
        Optional[Any]: The configured model instance or None if not configured
    """
    global _model
    return _model

def configure_gemini(api_key: str) -> Optional[Any]:
    """Configure Gemini AI model with the provided API key.
    
    Args:
        api_key (str): The Gemini API key
        
    Returns:
        Optional[Any]: The configured model instance or None if configuration failed
    """
    global _model
    try:
        genai.configure(api_key=api_key)
        
        # List available models
        print("\nAvailable Gemini Models:")
        print("------------------------")
        for m in genai.list_models():
            print(f"- Name: {m.name}")
            print(f"  Supported Generation Methods: {', '.join(m.supported_generation_methods)}")
            print(f"  Display Name: {m.display_name}")
            print(f"  Description: {m.description}")
            print("  ---")
        
        # Initialize model - using fastest stable model
        MODEL_NAME = "models/gemini-2.5-flash"  # Fast and efficient model, stable as of June 2025
        _model = genai.GenerativeModel(MODEL_NAME)
        print(f"\nSuccessfully initialized Gemini model: {MODEL_NAME} (optimized for speed)")
        return _model
    except Exception as e:
        print(f"\nError configuring Gemini: {str(e)}")
        _model = None
        return None

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
        print(f"Video {video_id} is unavailable")
        return []
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video {video_id}")
        return []
    except Exception as e:
        print(f"Error fetching transcripts for video {video_id}: {str(e)}")
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
        
        print("\nDEBUG: Listing available transcripts:")
        # Get list of all transcripts 
        for transcript in transcript_list:
            print("Found transcript:")
            print(f"- Language: {transcript.language} ({transcript.language_code})")
            print(f"- Generated: {transcript.is_generated}")
            print(f"- Translatable: {transcript.is_translatable}")
            
            # Store first transcript we see as fallback
            if first_transcript is None:
                first_transcript = transcript
            
            if not transcript.is_generated:
                # Found a manual transcript, use it
                print("Found manual transcript - using it")
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
                    print(f"Error fetching manual transcript: {str(e)}")
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
            print(f"\nUsing first available transcript: {first_transcript.language} ({first_transcript.language_code})")
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
                print(f"Error fetching first available transcript: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to fetch transcript: {str(e)}"
                )
            
        # If we get here, no transcript was found or all fetches failed
        raise HTTPException(
            status_code=404,
            detail={
                "message": "No transcript found",
                "available_transcripts": available_transcripts
            }
        )
            
    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Transcripts are disabled for this video")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable")
    except Exception as e:
        print(f"Error in get_best_transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching transcript: {str(e)}")

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
            print(f"WARNING: Failed to process transcript segment: {str(e)}")
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
    print(f"\nDEBUG: Processing transcript data of type: {type(transcript_data)}")
    
    # Handle FetchedTranscript object from youtube_transcript_api
    if str(type(transcript_data)) == "<class 'youtube_transcript_api._transcripts.FetchedTranscript'>":
        print("DEBUG: Input is FetchedTranscript from youtube_transcript_api")
        try:
            print(f"DEBUG: Available attributes: {dir(transcript_data)}")
            
            # Try to access the transcript segments directly
            segments = []
            
            # Try to access raw property if available
            if hasattr(transcript_data, '_transcripts') and isinstance(transcript_data._transcripts, list):
                print("DEBUG: Found _transcripts as list")
                segments = transcript_data._transcripts
            elif hasattr(transcript_data, 'snippets'):
                print("DEBUG: Found snippets attribute")
                segments = transcript_data.snippets
            elif str(transcript_data).startswith('FetchedTranscript(snippets=['):
                print("DEBUG: Parsing str representation")
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
                print(f"DEBUG: Found {len(segments)} segments")
                transcript_data = segments
            else:
                print("WARNING: Could not extract segments, using empty list")
                transcript_data = []
        except Exception as e:
            print(f"ERROR: Failed to extract from FetchedTranscript: {str(e)}")
            print(f"Object details: {dir(transcript_data)}")
            raise ValueError(f"Failed to process FetchedTranscript: {str(e)}")
    
    # Convert single dict/object to list
    if isinstance(transcript_data, dict) or hasattr(transcript_data, 'text'):
        print("DEBUG: Converting single segment to list")
        transcript_data = [transcript_data]
    
    # Ensure we have a list
    if not isinstance(transcript_data, list):
        print(f"ERROR: Expected list or FetchedTranscript, got {type(transcript_data)}")
        raise ValueError(f"Expected list or FetchedTranscript input, got {type(transcript_data)}")
    
    transcript_text = []
    try:
        print(f"DEBUG: Processing {len(transcript_data)} segments")
        for segment in transcript_data:
            text = None
            print(f"DEBUG: Processing segment of type: {type(segment)}")
            print(f"DEBUG: Segment content: {segment}")
            
            # First try the object interface (FetchedTranscriptSnippet)
            if hasattr(segment, 'text'):
                try:
                    text = getattr(segment, 'text')
                    print(f"DEBUG: Extracted text using attribute: {text[:50]}...")
                except AttributeError as ae:
                    print(f"WARNING: Failed to access text attribute: {str(ae)}")
                    print(f"Available attributes: {dir(segment)}")
                    continue
            
            # Try dict interface with fallbacks
            elif isinstance(segment, dict):
                # Try common key variations
                keys_to_try = ['text', 'content', 'transcript', 'value']
                for key in keys_to_try:
                    text = segment.get(key)
                    if text is not None:
                        print(f"DEBUG: Found text using key '{key}': {text[:50]}...")
                        break
                if text is None:
                    print(f"WARNING: Could not find text in dict with keys: {list(segment.keys())}")
                    continue
                    
            else:
                try:
                    # Last resort - try string conversion
                    text = str(segment).strip()
                    if text and not text.startswith('<') and not text.endswith('>'):
                        print(f"DEBUG: Converted segment to string: {text[:50]}...")
                    else:
                        print("WARNING: Segment conversion yielded invalid text")
                        continue
                except Exception as e:
                    print(f"WARNING: Could not process segment of type {type(segment)}: {str(e)}")
                    continue
                
            # Ensure we have text
            if text is None:
                continue
                
            # Convert to string safely
            try:
                text_str = str(text).strip()
                if text_str:  # Only append non-empty strings
                    transcript_text.append(text_str)
                    print(f"DEBUG: Added text segment: {text_str[:50]}...")
            except Exception as e:
                print(f"WARNING: Failed to convert text to string: {str(e)}")
                print(f"Problematic text content: {text}")
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
        print(f"Error extracting text from transcript: {str(e)}")
        print(f"Full error context: {str(e.__class__.__name__)}: {str(e)}")
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
            print("WARNING: Generated text appears to be missing content")
            return text  # Return original if too much content was lost
            
        return result
    except Exception as e:
        print(f"Error formatting transcript: {str(e)}")
        return text  # Return original text as fallback
