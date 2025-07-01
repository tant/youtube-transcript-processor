"""
FastAPI service for extracting and processing YouTube video transcripts.
Supports smart transcript selection, AI-powered formatting, and real-time streaming.

Key features:
- Smart transcript selection (manual preferred over auto-generated)
- Cleaning and formatting with Gemini AI
- Real-time streaming with SSE
- Full language support with Unicode
- Error handling with detailed responses

Environment variables:
    GEMINI_API_KEY: Required for AI features
    PORT: Optional server port (default: 8000)
    HOST: Optional host address (default: 0.0.0.0)
    ENVIRONMENT: Optional environment name (development/production)
"""

from fastapi import FastAPI, HTTPException
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    VideoUnavailable
)
from dotenv import load_dotenv
import os
import re
import logging
from .utils.transcript_helpers import (
    configure_gemini,
    extract_video_id,
    generate_short_summary,
    get_available_transcripts,
    get_best_transcript,
    extract_transcript_segments,
    get_model,
    get_video_info,
    clean_transcript_text
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="YouTube Transcript Extractor",
    description=__doc__,
    version="1.0.0"
)

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
else:
    configure_gemini(api_key)

@app.get("/raw-transcript/{video_id}")
async def get_raw_transcript(video_id: str):
    """Get the raw transcript from a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        dict: Contains video_id and transcript as a list of text segments
        
    Raises:
        404: When transcript is not found or video is unavailable
        500: For other errors
    """
    try:
        # First try to get manual English transcript
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        except NoTranscriptFound:
            # Then try auto-generated English transcript
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en-GB', 'a.en'])
            except NoTranscriptFound:
                # Finally try to find any available transcript
                transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript_list = transcripts.find_transcript(['en', 'vi']).fetch()

        # Extract only the text from transcript using helper function
        try:
            transcript_text = extract_transcript_segments(transcript_list)
            if not isinstance(transcript_text, list):
                raise ValueError("extract_transcript_segments returned non-list result")
            
            if not transcript_text:
                raise HTTPException(status_code=404, detail="No valid text found in transcript")
        except Exception as e:
            logger.error("Error extracting transcript text: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process transcript: {str(e)}"
            )
                
        available = get_available_transcripts(video_id)
        video_info = get_video_info(video_id)
        
        return {
            "video_id": video_id,
            "video_info": video_info,
            "transcript": transcript_text,
            "available_transcripts": available
        }
    except NoTranscriptFound:
        # If no transcript found, return list of available ones if any
        available = get_available_transcripts(video_id)
        if available:
            raise HTTPException(
                status_code=404, 
                detail={
                    "message": "No English transcript found",
                    "available_transcripts": available
                }
            )
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transcript: {str(e)}")

@app.get("/to-url/{video_id}")
async def get_full_url(video_id: str):
    """Convert a YouTube video ID to its full URL.
    
    Args:
        video_id (str): The YouTube video ID (11 characters)
        
    Returns:
        dict: Contains video_id and the full YouTube URL
        
    Raises:
        400: When video ID format is invalid
    """
    if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}"
    }

@app.get("/to-id/")
async def get_video_id(url: str):
    """Extract video ID from a YouTube URL.
    
    Args:
        url (str): The YouTube URL or video ID
        
    Returns:
        dict: Contains extracted video_id and the original URL
        
    Raises:
        400: When URL is invalid or video ID cannot be extracted
    """
    # Kiểm tra xem có phải URL YouTube không
    if not any(domain in url.lower() for domain in ['youtube.com', 'youtu.be']) and len(url) != 11:
        raise HTTPException(status_code=400, detail="Could not extract video ID from URL")
    
    video_id = extract_video_id(url)
    if video_id is None:
        raise HTTPException(status_code=400, detail="Could not extract video ID from URL")
    return {
        "video_id": video_id,
        "url": url
    }

@app.get("/clean-transcript/{video_id}")
async def get_clean_transcript(video_id: str):
    """Get a cleaned and AI-enhanced transcript from a YouTube video.
    
    Processing steps:
    1. Smart transcript selection:
       - Tries manual English transcript first
       - Falls back to auto-generated English
       - Finally tries any available language
    2. Cleaning:
       - Removes sound descriptions ([Music], [Applause], etc.)
       - Strips extra whitespace and newlines
       - Validates text segments
    3. AI enhancement with Gemini:
       - Reconstructs text into proper paragraphs
       - Maintains original language and formatting
       - Generates concise English summary
    
    Args:
        video_id (str): The YouTube video ID to process
        
    Returns:
        dict: A JSON response containing:
            video_id (str): The original video ID
            video_info (dict): Metadata about the video (title, author, etc.)
            transcript (str): Cleaned and formatted text
            segments_count (int): Number of original segments
            short_summary (str): 100-word summary in English
    
    Note:
        - Preserves original language while providing English summary
        - Handles Unicode characters properly
        - Uses Gemini AI for text reconstruction and summarization
    """
    try:
        transcript_list, transcript_info = get_best_transcript(video_id)
        
        if not transcript_list:
            raise HTTPException(status_code=404, detail="No transcript found")
            
        # Extract text from transcript using helper function
        try:
            transcript_text = extract_transcript_segments(transcript_list)
            if not isinstance(transcript_text, list) or not transcript_text:
                raise ValueError("Transcript extraction resulted in invalid format or empty list")
        except Exception as e:
            logger.error("Error extracting transcript text: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract text from transcript: {str(e)}"
            )
                
        model = get_model()
        if not model:
            raise HTTPException(
                status_code=500, 
                detail="Gemini model not available. Check API key."
            )

        # Clean and reconstruct transcript
        cleaned_transcript = clean_transcript_text(transcript_text)
        
        # Generate a short summary
        summary = generate_short_summary(cleaned_transcript)

        video_info = get_video_info(video_id)

        return {
            "video_id": video_id,
            "video_info": video_info,
            "transcript": cleaned_transcript,
            "segments_count": len(transcript_list),
            "short_summary": summary,
            "transcript_info": transcript_info
        }
    except NoTranscriptFound:
        available = get_available_transcripts(video_id)
        if available:
            raise HTTPException(
                status_code=404, 
                detail={
                    "message": "No suitable transcript found for cleaning",
                    "available_transcripts": available
                }
            )
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable")
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_clean_transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/available-transcripts/{video_id}")
async def check_available_transcripts(video_id: str):
    """List all available transcripts for a video.
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        dict: Contains video_id and list of available transcript languages/types
        
    Raises:
        404: When video is unavailable
        500: For other errors
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        transcripts = []
        for transcript in transcript_list:
            transcripts.append({
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "is_translatable": transcript.is_translatable
            })
            
        return {
            "video_id": video_id,
            "available_transcripts": transcripts
        }
        
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-info/{video_id}")
async def get_video_metadata(video_id: str):
    """Get basic information about a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        dict: Video metadata including title, author, published date
        
    Raises:
        404: When video is unavailable
        500: For other errors
    """
    try:
        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
            raise HTTPException(status_code=400, detail="Invalid video ID format")
            
        video_info = get_video_info(video_id)
        if not video_info.get("title"):
            raise HTTPException(status_code=404, detail="Could not retrieve video information")
            
        return {
            "video_id": video_id,
            "video_info": video_info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching video info: {str(e)}")
