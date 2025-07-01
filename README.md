# YouTube Content Extractor

A FastAPI service that extracts and processes YouTube video transcripts with smart transcript selection and AI-powered formatting.

## Features

- Smart transcript selection with priority:
  1. Manual transcripts in any language
  2. Auto-generated transcripts as fallback
  - Always preserves original language
  - English summaries for all languages
- Get raw YouTube transcripts with language info
- Clean and format transcripts:
  - Removes sound descriptions ([Music], [Applause], etc.)
  - Joins transcript pieces into coherent paragraphs 
  - Returns formatted Markdown
  - Preserves Unicode characters (supports all languages)
- Real-time streaming processing with SSE:
  1. Raw transcript (immediate)
  2. Cleaned transcript (no sound effects)
  3. Formatted text (AI-reconstructed)
  4. Short summary
- Convert between video ID and URL formats
- List all available transcripts with full language info
- AI-powered features (requires Gemini API key):
  - Reconstruct transcript into proper paragraphs
  - Generate 100-word English summary
  - Handles multilingual content

## API Endpoints 

### GET /raw-transcript/{video_id}
Get the raw transcript for a YouTube video. Will attempt to find the best available transcript using smart selection.

Response:
```json
{
    "video_id": "...",
    "transcript": ["segment 1", "segment 2", ...],
    "available_transcripts": [
        {
            "language": "English",
            "language_code": "en",
            "is_generated": false,
            "is_translatable": true
        }
    ]
}
```

### GET /clean-transcript/{video_id}
Get a cleaned and formatted transcript with AI-generated summary. Supports any language while providing English summary.

Response:
```json
{
    "video_id": "...",
    "transcript": "Full formatted text in original language...",
    "segments_count": 42,
    "short_summary": "100-word English summary...",
    "transcript_info": {
        "language": "Vietnamese",
        "language_code": "vi",
        "type": "manual",
        "is_generated": false
    }
}
```

### GET /clean-transcript-stream/{video_id} 
Stream the transcript processing steps via Server-Sent Events (SSE). Each step includes transcript info and preserves Unicode.

Events:
1. raw: Original transcript segments
2. cleaned: Transcript with sound descriptions removed
3. complete: AI-formatted text in paragraphs
4. summary: Final text with 100-word summary

Example event:
```json
{
    "status": "complete",
    "transcript": "Formatted text in paragraphs...",
    "segments_count": 42,
    "transcript_info": {
        "language": "Japanese",
        "language_code": "ja",
        "type": "auto",
        "is_generated": true
    }
}
```

### GET /available-transcripts/{video_id}
List all available transcripts for a video with detailed language information.

Response:
```json
{
    "video_id": "...",
    "available_transcripts": [
        {
            "language": "English",
            "language_code": "en",
            "is_generated": false,
            "is_translatable": true
        },
        {
            "language": "Vietnamese",
            "language_code": "vi", 
            "is_generated": true,
            "is_translatable": false
        }
    ]
}
```

### GET /to-url/{video_id}
Convert a video ID to full YouTube URL.

Response:
```json
{
    "video_id": "dQw4w9WgXcQ",
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

### GET /to-id/?url={youtube_url}
Convert a YouTube URL to video ID.

Response:
```json
{
    "video_id": "dQw4w9WgXcQ",
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create .env file with Gemini API key:
```
GEMINI_API_KEY=your_key_here
```

3. Run the server:
```bash 
uvicorn app.main:app --reload
```

## Language Support

The service supports any language available in YouTube transcripts:
- Can use manual or auto-generated transcripts
- Preserves original language in transcript
- Always generates summary in English
- Full Unicode support for all languages
- Smart transcript selection prioritizes:
  1. Manual transcripts (any language)
  2. Auto-generated transcripts as fallback

## Error Handling

- 404: Video not found or no transcript available
- 400: Invalid video ID or URL format
- 500: Server error or Gemini API issues

Detailed error responses include:
- Error message
- Available transcripts when relevant
- Transcript language info when available

## License

MIT
