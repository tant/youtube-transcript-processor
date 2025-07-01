from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound, 
    TranscriptsDisabled,
    VideoUnavailable
)
from dotenv import load_dotenv
import os
import re
import json
import google.generativeai as genai

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

def generate_short_summary(text: str) -> str:
    """Generate a short 100-word summary of the transcript using Gemini.
    
    Args:
        text (str): The full transcript text to summarize
        
    Returns:
        str: A 100-word summary of the content
        
    Raises:
        ValueError: If Gemini model is not configured or generation fails
    """
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

def get_available_transcripts(video_id: str) -> list:
    """Get list of available transcripts for a video.
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        list: List of available transcript languages and types
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
    except Exception as e:
        return []

def get_best_transcript(video_id: str):
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
            print(f"Found transcript:")
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

        # Extract only the text from transcript
        transcript_text = [item["text"] for item in transcript_list]
        
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
    """Get a cleaned and reconstructed transcript from a YouTube video using Gemini.
    
    This endpoint:
    1. Fetches the best available transcript (manual preferred over auto-generated)
    2. Removes sound descriptions ([Music], [Applause], etc.)
    3. Uses Gemini AI to reconstruct text into proper paragraphs
    4. Generates a 100-word summary in English
    5. Returns formatted result in Markdown
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        dict: Contains:
            - video_id: The original video ID
            - transcript: Cleaned and formatted text in Markdown 
            - segments_count: Number of original transcript segments
            - short_summary: A 100-word summary in English
            - transcript_info: Language and type information
        
    Raises:
        404: When transcript is not found or video is unavailable
        500: When Gemini is not configured or for other errors
    """
    try:
        # Get best available transcript
        print(f"\nGetting transcript for video {video_id}...")
        transcript_list, transcript_info = get_best_transcript(video_id)
        print("Found transcript:", transcript_info)
        
        if not transcript_list:
            raise HTTPException(status_code=404, detail="No transcript found")
            
        # Extract text from transcript
        transcript_text = []
        try:
            for segment in transcript_list:
                # YouTubeTranscriptApi returns dict with 'text', 'start', 'duration'
                if hasattr(segment, 'text'):
                    # Object with text attribute
                    transcript_text.append(segment.text)
                elif isinstance(segment, dict):
                    # Dictionary with text key
                    if 'text' in segment:
                        transcript_text.append(segment['text'])
                    else:
                        print(f"WARNING: Dict segment missing 'text' key: {segment}")
                else:
                    print(f"WARNING: Unexpected transcript segment type: {type(segment)}")
                    
        except Exception as e:
            print(f"Error extracting text from transcript: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract text from transcript: {str(e)}"
            )
                
        print(f"Extracted {len(transcript_text)} segments")
        
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
    """Stream the transcript cleaning process for a YouTube video.
    
    This endpoint streams the process in steps:
    1. Returns raw transcript immediately
    2. Updates with cleaned transcript (removed sound descriptions)
    3. Sends the Gemini-processed formatted text
    4. Finally returns a 100-word summary
    
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
            # Step 1: Get best available transcript
            print(f"\nGetting transcript for video {video_id}...")
            transcript_list, transcript_info = get_best_transcript(video_id)
            print("Found transcript:", transcript_info)
            
            if not transcript_list:
                yield f"data: {json.dumps({'status': 'error', 'detail': 'No transcript found'}, ensure_ascii=False)}\n\n"
                return
                
            # Extract text from transcript segments
            transcript_text = []
            try:
                for segment in transcript_list:
                    # YouTubeTranscriptApi returns dict with 'text', 'start', 'duration'
                    if hasattr(segment, 'text'):
                        # Object with text attribute
                        transcript_text.append(segment.text)
                    elif isinstance(segment, dict):
                        # Dictionary with text key
                        if 'text' in segment:
                            transcript_text.append(segment['text'])
                        else:
                            print(f"WARNING: Dict segment missing 'text' key: {segment}")
                    else:
                        print(f"WARNING: Unexpected transcript segment type: {type(segment)}")
                        
            except Exception as e:
                print(f"Error extracting text from transcript: {str(e)}")
                yield f"data: {json.dumps({'status': 'error', 'detail': f'Failed to extract text from transcript: {str(e)}'}, ensure_ascii=False)}\n\n"
                return
                    
            print(f"Extracted {len(transcript_text)} segments")
            
            # Send raw transcript with language info
            yield f"data: {json.dumps({
                'status': 'raw',
                'transcript': transcript_text,
                'transcript_info': transcript_info
            }, ensure_ascii=False)}\n\n"
            
            if not model:
                yield f"data: {json.dumps({'status': 'error', 'detail': 'Gemini model not properly configured'}, ensure_ascii=False)}\n\n"
                return

            if not transcript_text:
                yield f"data: {json.dumps({'status': 'error', 'detail': 'Empty transcript received'}, ensure_ascii=False)}\n\n"
                return

            # Step 2: Clean transcript
            cleaned_text = []
            sound_patterns = [r'\[Music\]', r'\[Applause\]', r'\[.*?\]']
            
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
                yield f"data: {json.dumps({'status': 'error', 'detail': 'All transcript segments were empty after cleaning'}, ensure_ascii=False)}\n\n"
                return
            
            # Send cleaned transcript
            yield f"data: {json.dumps({
                'status': 'cleaned',
                'transcript': cleaned_text,
                'segments_count': len(transcript_text),
                'transcript_info': transcript_info
            }, ensure_ascii=False)}\n\n"
            
            # Step 3: Process with Gemini for proper formatting
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

Segments to reconstruct:
{}

Output the reconstructed text in Markdown format with proper paragraph breaks.""".format(" ".join(cleaned_text))

            try:
                response = model.generate_content(prompt)
                reconstructed_text = response.text.strip()
                
                # Validate output length
                total_input_length = sum(len(s) for s in transcript_text)
                if len(reconstructed_text) < total_input_length * 0.9:
                    print("WARNING: Generated transcript appears to be missing content")
                    # Use cleaned text as fallback
                    reconstructed_text = ' '.join(cleaned_text)
            except Exception as e:
                print(f"Error generating formatted text: {str(e)}")
                # Use cleaned text as fallback
                reconstructed_text = ' '.join(cleaned_text)
            
            # Send formatted transcript
            yield f"data: {json.dumps({
                'status': 'complete',
                'transcript': reconstructed_text,
                'segments_count': len(transcript_text),
                'transcript_info': transcript_info
            }, ensure_ascii=False)}\n\n"
            
            # Step 4: Generate and send summary
            try:
                short_summary = generate_short_summary(reconstructed_text)
                yield f"data: {json.dumps({
                    'status': 'summary',
                    'transcript': reconstructed_text,
                    'segments_count': len(transcript_text),
                    'short_summary': short_summary,
                    'transcript_info': transcript_info
                }, ensure_ascii=False)}\n\n"
            except Exception as e:
                print(f"Failed to generate summary: {str(e)}")
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
