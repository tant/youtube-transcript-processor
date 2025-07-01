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
from .db import Base, SessionLocal, init_db
from sqlalchemy import Column, String, Text

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

# Update the Transcript table schema
class Transcript(Base):
    __tablename__ = "transcripts"
    video_id = Column(String, primary_key=True, index=True)
    raw_transcript = Column(Text, nullable=True)  # Store raw transcript
    clean_transcript = Column(Text, nullable=True)  # Store cleaned transcript
    video_info = Column(Text, nullable=True)

@app.on_event("startup")
def startup_event():
    """Initialize the database and create tables on application startup if not already created."""
    try:
        from sqlalchemy.engine import reflection
        inspector = reflection.Inspector.from_engine(SessionLocal().bind)
        if "transcripts" not in inspector.get_table_names():
            init_db()
            logger.info("Database initialized and tables created successfully.")
        else:
            logger.info("Database tables already exist. Skipping initialization.")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
else:
    configure_gemini(api_key)

@app.get("/raw-transcript/{video_id}")
async def get_raw_transcript(video_id: str, cache: bool = True):
    """Get the raw transcript from a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID
        cache (bool): Whether to save the result to the database (default: True)
    
    Returns:
        dict: Contains video_id and transcript as a list of text segments
        
    Raises:
        404: When transcript is not found or video is unavailable
        500: For other errors
    """
    db = SessionLocal()
    existing_transcript = db.query(Transcript).filter(Transcript.video_id == video_id).first()
    if existing_transcript and existing_transcript.raw_transcript:
        return {
            "video_id": video_id,
            "video_info": existing_transcript.video_info,
            "transcript": existing_transcript.raw_transcript  # Already stored as a string
        }
    elif existing_transcript and not existing_transcript.raw_transcript:
        logger.info(f"Transcript found in DB but empty, fetching new transcript for video_id: {video_id}")

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
        
        # Convert list to string before saving to database
        transcript_text_str = '\n'.join(transcript_text)
        result = {
            "video_id": video_id,
            "video_info": video_info,
            "transcript": transcript_text_str,  # Return as a single string
            "available_transcripts": available
        }

        # Save to database only if cache is True
        if cache:
            if existing_transcript:
                existing_transcript.raw_transcript = transcript_text_str
                existing_transcript.video_info = str(video_info)
            else:
                new_transcript = Transcript(
                    video_id=video_id,
                    raw_transcript=transcript_text_str,
                    video_info=str(video_info)
                )
                db.add(new_transcript)
            db.commit()
        return result
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

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
async def get_clean_transcript(video_id: str, cache: bool = True):
    """Get a cleaned and AI-enhanced transcript from a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID
        cache (bool): Whether to save the result to the database (default: True)
    
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
    db = SessionLocal()
    existing_transcript = db.query(Transcript).filter(Transcript.video_id == video_id).first()

    if existing_transcript:
        if existing_transcript.clean_transcript:
            return {
                "video_id": video_id,
                "video_info": existing_transcript.video_info,
                "transcript": existing_transcript.clean_transcript
            }
        elif existing_transcript.raw_transcript:
            # Use the existing raw_transcript to clean and process
            transcript_text = existing_transcript.raw_transcript.split('\n')
        else:
            # Fetch new transcript if raw_transcript is not available
            transcript_list, transcript_info = get_best_transcript(video_id)
            if not transcript_list:
                raise HTTPException(status_code=404, detail="No transcript found")

            transcript_text = extract_transcript_segments(transcript_list)
            if not isinstance(transcript_text, list) or not transcript_text:
                raise ValueError("Transcript extraction resulted in invalid format or empty list")
    else:
        # Fetch new transcript if video_id does not exist in the database
        transcript_list, transcript_info = get_best_transcript(video_id)
        if not transcript_list:
            raise HTTPException(status_code=404, detail="No transcript found")

        transcript_text = extract_transcript_segments(transcript_list)
        if not isinstance(transcript_text, list) or not transcript_text:
            raise ValueError("Transcript extraction resulted in invalid format or empty list")

    try:
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

        result = {
            "video_id": video_id,
            "video_info": video_info,
            "transcript": cleaned_transcript,
            "segments_count": len(transcript_text),
            "short_summary": summary
        }

        # Save to database only if cache is True
        if cache:
            if existing_transcript:
                existing_transcript.clean_transcript = cleaned_transcript
                existing_transcript.video_info = str(video_info)
            else:
                new_transcript = Transcript(
                    video_id=video_id,
                    clean_transcript=cleaned_transcript,
                    video_info=str(video_info)
                )
                db.add(new_transcript)
            db.commit()
        return result
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

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
