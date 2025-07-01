from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import json
import json
from fastapi.responses import StreamingResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="YouTube Transcript Extractor")

# Configure Gemini
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("\nERROR: GEMINI_API_KEY not found in environment variables")
    model = None
else:
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
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"\nSuccessfully initialized Gemini model: {MODEL_NAME} (optimized for speed)")
    except Exception as e:
        print(f"\nError configuring Gemini: {str(e)}")
        model = None

def extract_video_id(url: str) -> str:
    """Extract video ID from various YouTube URL formats."""
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
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Extract only the text from transcript
        transcript_text = [item["text"] for item in transcript_list]
        return {
            "video_id": video_id,
            "transcript": transcript_text
        }
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    """Get a cleaned and reconstructed transcript from a YouTube video using Gemini.
    
    This endpoint:
    1. Fetches the raw transcript
    2. Removes sound descriptions ([Music], [Applause], etc.)
    3. Uses Gemini AI to reconstruct text into proper paragraphs
    4. Returns formatted result in Markdown
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        dict: Contains:
            - video_id: The original video ID
            - transcript: Cleaned and formatted text in Markdown
            - segments_count: Number of original transcript segments
        
    Raises:
        404: When transcript is not found or video is unavailable
        500: When Gemini is not configured or for other errors
    """
    try:
        # First get the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Extract only the text
        transcript_text = [item["text"] for item in transcript_list]
        
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
            # Remove sound descriptions
            clean_text = text
            for pattern in sound_patterns:
                clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
            
            # Remove newlines and extra spaces
            clean_text = clean_text.replace('\n', ' ').strip()
            
            # Only add non-empty segments
            if clean_text:
                cleaned_text.append(clean_text)
        
        # Prompt Gemini Flash model with concise instructions
        prompt = """Reconstruct these transcript segments into a well-formatted text. Rules:
- Keep 100% of original words
- Only fix: connections between segments, spacing, and punctuation
- No summarizing, no new content
- Group related content into logical paragraphs
- Format output as Markdown with proper paragraphs
- Use two newlines between paragraphs for proper Markdown formatting
- For any speaker changes or major topic shifts, start a new paragraph
- Preserve any quotes or important phrases exactly as they appear

Segments to reconstruct:
{}

Output the reconstructed text in Markdown format with proper paragraph breaks.""".format(" ".join(cleaned_text))
        
        try:
            response = model.generate_content(prompt)
            reconstructed_text = response.text.strip()
            
            # Validate output length is reasonable compared to input
            total_input_length = sum(len(s) for s in transcript_text)
            if len(reconstructed_text) < total_input_length * 0.9:  # Allow for some punctuation/spacing optimization
                raise ValueError("Generated transcript appears to be missing content")
                
            return {
                "video_id": video_id,
                "transcript": reconstructed_text,
                "segments_count": len(transcript_text)  # Additional info for validation
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating coherent transcript: {str(e)}"
            )
        
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clean-transcript-stream/{video_id}")
async def get_clean_transcript_stream(video_id: str):
    """Stream the transcript cleaning process for a YouTube video.
    
    This endpoint streams the process in steps:
    1. Returns raw transcript immediately
    2. Updates with cleaned transcript (removed sound descriptions)
    3. Finally returns the Gemini-processed formatted text
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        StreamingResponse: Server-sent events with transcript processing updates
        
    Raises:
        404: When transcript is not found or video is unavailable
        500: When Gemini is not configured or for other errors
    """
    async def generate():
        try:
            # Step 1: Get raw transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = [item["text"] for item in transcript_list]
            
            # Send raw transcript immediately
            yield f"data: {json.dumps({'status': 'raw', 'transcript': transcript_text})}\n\n"
            
            if not model:
                raise HTTPException(
                    status_code=500,
                    detail="Gemini model not properly configured. Please check your API key."
                )

            # Step 2: Clean transcript
            cleaned_text = []
            sound_patterns = [r'\[Music\]', r'\[Applause\]', r'\[.*?\]']
            
            for text in transcript_text:
                clean_text = text
                for pattern in sound_patterns:
                    clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
                clean_text = clean_text.replace('\n', ' ').strip()
                if clean_text:
                    cleaned_text.append(clean_text)
            
            # Send cleaned transcript
            yield f"data: {json.dumps({'status': 'cleaned', 'transcript': cleaned_text})}\n\n"
            
            # Step 3: Process with Gemini
            prompt = """Reconstruct these transcript segments into a well-formatted text. Rules:
- Keep 100% of original words
- Only fix: connections between segments, spacing, and punctuation
- No summarizing, no new content
- Group related content into logical paragraphs
- Format output as Markdown with proper paragraphs
- Use two newlines between paragraphs for proper Markdown formatting
- For any speaker changes or major topic shifts, start a new paragraph
- Preserve any quotes or important phrases exactly as they appear

Segments to reconstruct:
{}

Output the reconstructed text in Markdown format with proper paragraph breaks.""".format(" ".join(cleaned_text))

            response = model.generate_content(prompt)
            reconstructed_text = response.text.strip()
            
            # Validate and send final result
            total_input_length = sum(len(s) for s in transcript_text)
            if len(reconstructed_text) < total_input_length * 0.9:
                raise ValueError("Generated transcript appears to be missing content")
                
            yield f"data: {json.dumps({'status': 'complete', 'transcript': reconstructed_text, 'segments_count': len(transcript_text)})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'detail': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
