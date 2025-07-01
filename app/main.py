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
from fastapi.responses import StreamingResponse
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    VideoUnavailable
    # TranscriptsDisabled is not used
)
from dotenv import load_dotenv
import os
import re
import json
from .utils.transcript_helpers import (
    configure_gemini,
    extract_video_id,
    generate_short_summary,
    get_available_transcripts,
    get_best_transcript,
    clean_transcript_text,
    extract_transcript_segments,
    format_transcript_with_gemini,
    get_model
)

# Load and verify environment configuration
load_dotenv()

# Initialize FastAPI with metadata
app = FastAPI(
    title="YouTube Transcript Extractor",
    description=__doc__,
    version="1.0.0"
)

# Debug environment variables to verify configuration
print("\nDEBUG: Environment variables loaded:")
print(f"PORT: {os.getenv('PORT')}")
print(f"HOST: {os.getenv('HOST')}")
print(f"ENVIRONMENT: {os.getenv('ENVIRONMENT')}")
print(f"GEMINI_API_KEY exists: {bool(os.getenv('GEMINI_API_KEY'))}")
print(f"GEMINI_API_KEY length: {len(os.getenv('GEMINI_API_KEY') or '')}")

# Configure Gemini AI with API key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("\nERROR: GEMINI_API_KEY not found in environment variables")
else:
    print(f"\nINFO: Configuring Gemini with API key of length {len(api_key)}")
    configure_gemini(api_key)

# Initialize FastAPI app
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
            print(f"Error extracting transcript text: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process transcript: {str(e)}"
            )
                
        # Get available transcripts for info
        available = get_available_transcripts(video_id)
        
        return {
            "video_id": video_id,
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
            transcript (str): Cleaned and formatted text
            segments_count (int): Number of original segments
            short_summary (str): 100-word summary in English
            transcript_info (dict): Language and source info:
                language (str): Full language name
                language_code (str): ISO code
                type (str): 'manual' or 'auto'
                is_generated (bool): Auto-generated flag
        
    Raises:
        HTTPException(404): Video not found or no transcript
        HTTPException(500): Gemini error or processing failure
    
    Note:
        - Preserves original language while providing English summary
        - Handles Unicode characters properly
        - Uses Gemini AI for text reconstruction and summarization
    """
    try:
        # Get best available transcript
        print(f"\nGetting transcript for video {video_id}...")
        transcript_list, transcript_info = get_best_transcript(video_id)
        print("Found transcript:", transcript_info)
        
        if not transcript_list:
            raise HTTPException(status_code=404, detail="No transcript found")
            
        # Extract text from transcript using helper function
        try:
            transcript_text = extract_transcript_segments(transcript_list)
            if not isinstance(transcript_text, list):
                raise ValueError("extract_transcript_segments returned non-list result")
                
            # Validate all segments are strings
            for segment in transcript_text:
                if not isinstance(segment, str):
                    print(f"WARNING: Non-string transcript text: {type(segment)}")
                    transcript_text = [str(s) for s in transcript_text if s is not None]
                    break
                    
        except Exception as e:
            print(f"Error extracting text from transcript: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract text from transcript: {str(e)}"
            )
                
        print(f"Extracted {len(transcript_text)} segments")
        
        model = get_model()
        if not model:
            raise HTTPException(
                status_code=500,
                detail="Gemini model not properly configured. Please check your API key."
            )

        if not transcript_text:
            raise HTTPException(
                status_code=404,
                detail="Empty transcript received from video"
            )

        # Clean transcript text: remove sound descriptions and newlines
        cleaned_text = []
        sound_patterns = [r'\[Music\]', r'\[Applause\]', r'\[.*?\]']  # Add more patterns as needed
        
        for text in transcript_text:
            if not isinstance(text, str):
                print(f"WARNING: Non-string transcript text: {type(text)}")
                continue
                
            # Remove sound descriptions
            clean_text = text
            for pattern in sound_patterns:
                clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
            
            # Remove newlines and extra spaces
            clean_text = clean_text.replace('\n', ' ').strip()
            
            # Add if not empty
            if clean_text:
                cleaned_text.append(clean_text)
                
        if not cleaned_text:
            raise HTTPException(
                status_code=500,
                detail="All transcript segments were empty after cleaning"
            )
                
        # Join cleaned text
        complete_text = ' '.join(cleaned_text)
        
        try:
            # Generate short summary using Gemini
            short_summary = generate_short_summary(complete_text)
        except Exception as e:
            print(f"Failed to generate summary: {str(e)}")
            short_summary = None

        # Return response
        response = {
            "video_id": video_id,
            "transcript": complete_text,
            "segments_count": len(transcript_text),
            "short_summary": short_summary,
            "transcript_info": transcript_info
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clean-transcript-stream/{video_id}")
async def get_clean_transcript_stream(video_id: str):
    """Stream the transcript cleaning process for a YouTube video in real-time.
    
    This endpoint uses Server-Sent Events (SSE) to stream the processing steps:
    1. Raw transcript: Returns original segments immediately
    2. Cleaned transcript: Updates after removing sound descriptions
    3. Formatted text: Sends Gemini-processed, properly paragraphed text
    4. Summary: Finally returns a 100-word English summary
    
    Each event includes:
    - status: 'raw', 'cleaned', 'complete', 'summary', or 'error'
    - transcript: Text at current processing stage
    - transcript_info: Language and type information
    - segments_count: Number of original segments
    - short_summary: Only in final 'summary' event
    
    All events preserve Unicode characters and original language.
    
    Args:
        video_id (str): The YouTube video ID to process
        
    Returns:
        StreamingResponse: Server-sent events with processing updates.
        Each event is a JSON object with the structure:
        {
            "status": str,
            "transcript": str | list[str],
            "transcript_info": dict,
            "segments_count": int,
            "short_summary": str | None  # Only in summary event
        }
        
    Raises:
        404: When transcript is not found or video is unavailable
        500: When Gemini is not configured or other processing errors occur
    """
    async def generate():
        try:
            # Step 1: Smart transcript selection and initial fetch
            print(f"\nGetting transcript for video {video_id}...")
            transcript_list, transcript_info = get_best_transcript(video_id)
            print("Found transcript:", transcript_info)
            
            if not transcript_list:
                yield f"data: {json.dumps({'status': 'error', 'detail': 'No transcript found'}, ensure_ascii=False)}\n\n"
                return
                
            # Extract raw text segments, preserving original formatting
            try:
                transcript_text = extract_transcript_segments(transcript_list)
            except HTTPException as e:
                yield f"data: {json.dumps({'status': 'error', 'detail': e.detail}, ensure_ascii=False)}\n\n"
                return
                    
            print(f"Extracted {len(transcript_text)} segments")
            
            # Stream Event 1: Raw transcript with language info
            yield f"data: {json.dumps({
                'status': 'raw',
                'transcript': transcript_text,
                'transcript_info': transcript_info
            }, ensure_ascii=False)}\n\n"
            
            # Verify Gemini model before proceeding
            model = get_model()
            if not model:
                yield f"data: {json.dumps({'status': 'error', 'detail': 'Gemini model not properly configured'}, ensure_ascii=False)}\n\n"
                return

            if not transcript_text:
                yield f"data: {json.dumps({'status': 'error', 'detail': 'Empty transcript received'}, ensure_ascii=False)}\n\n"
                return

            # Step 2: Clean and normalize text
            cleaned_text = clean_transcript_text(transcript_text)
                    
            if not cleaned_text:
                yield f"data: {json.dumps({'status': 'error', 'detail': 'All transcript segments were empty after cleaning'}, ensure_ascii=False)}\n\n"
                return
            
            # Stream Event 2: Cleaned transcript without sound descriptions
            yield f"data: {json.dumps({
                'status': 'cleaned',
                'transcript': cleaned_text,
                'segments_count': len(transcript_text),
                'transcript_info': transcript_info
            }, ensure_ascii=False)}\n\n"
            
            # Step 3: Use Gemini to format into proper paragraphs
            reconstructed_text = format_transcript_with_gemini(cleaned_text)
            
            # Stream Event 3: Formatted text in paragraphs
            yield f"data: {json.dumps({
                'status': 'complete',
                'transcript': reconstructed_text,
                'segments_count': len(transcript_text),
                'transcript_info': transcript_info
            }, ensure_ascii=False)}\n\n"
            
            # Step 4: Generate and stream final summary
            try:
                # Create English summary while preserving original transcript
                short_summary = generate_short_summary(reconstructed_text)
                
                # Stream Event 4: Complete transcript with summary
                yield f"data: {json.dumps({
                    'status': 'summary',
                    'transcript': reconstructed_text,
                    'segments_count': len(transcript_text),
                    'short_summary': short_summary,
                    'transcript_info': transcript_info
                }, ensure_ascii=False)}\n\n"
            except Exception as e:
                print(f"Failed to generate summary: {str(e)}")
                # Still send final event but without summary
                yield f"data: {json.dumps({
                    'status': 'summary',
                    'transcript': reconstructed_text,
                    'segments_count': len(transcript_text),
                    'short_summary': None,
                    'transcript_info': transcript_info
                }, ensure_ascii=False)}\n\n"
            
        except HTTPException as he:
            yield f"data: {json.dumps({'status': 'error', 'detail': he.detail}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"Error processing transcript stream: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"
    
    # Return SSE response with proper content type
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/available-transcripts/{video_id}")
async def list_available_transcripts(video_id: str):
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
        
        # Get all available transcripts
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
