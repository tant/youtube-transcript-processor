# YouTube Transcript Extractor API

A FastAPI-based REST API that extracts and processes transcripts from YouTube videos, featuring AI-powered text cleanup and formatting using Google's Gemini model.

## Features

- Extract raw transcripts from YouTube videos
- Clean and format transcripts using Google's Gemini AI
- Remove sound descriptions ([Music], [Applause], etc.)
- Convert between video IDs and URLs
- FastAPI-powered REST API with automatic OpenAPI documentation
- Comprehensive error handling
- Markdown-formatted output

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd yt-content-extractor
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:

On Linux/macOS:
```bash
source .venv/bin/activate
```

On Windows:
```bash
.venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up environment variables:
```bash
cp .env.sample .env
```
Edit the `.env` file with your preferred configuration.

## Environment Variables

The following environment variables can be configured in your `.env` file:

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `ENVIRONMENT`: Application environment (development/production/testing)
- `RATE_LIMIT_PER_MINUTE`: API rate limit per minute
- `CACHE_ENABLED`: Enable/disable caching
- `CACHE_TTL`: Cache time-to-live in seconds
- `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins
- `API_KEY`: API key for authentication (if enabled)
- `GEMINI_API_KEY`: Your Google Gemini API key for AI features
- `DATABASE_URL`: Database connection string (if needed)

### Getting a Gemini API Key

To use AI features powered by Google's Gemini model:

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or sign in to your Google account
3. Click "Create API Key"
4. Copy the generated API key
5. Add it to your `.env` file:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

Note: Keep your API key secure and never commit it to version control.

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### GET /raw-transcript/{video_id}
Get the raw transcript segments from a YouTube video.

**Parameters:**
- `video_id`: YouTube video ID (string, required)

**Response:**
```json
{
    "video_id": "string",
    "transcript": ["string"]
}
```

### GET /clean-transcript/{video_id}
Get a cleaned and nicely formatted transcript using Gemini AI.

**Parameters:**
- `video_id`: YouTube video ID (string, required)

**Response:**
```json
{
    "video_id": "string",
    "transcript": "string (Markdown formatted)",
    "segments_count": number
}
```

### GET /to-url/{video_id}
Convert a video ID to full YouTube URL.

**Parameters:**
- `video_id`: YouTube video ID (string, required)

**Response:**
```json
{
    "video_id": "string",
    "url": "string"
}
```

### GET /to-id
Extract video ID from a YouTube URL.

**Parameters:**
- `url`: YouTube URL (string, query parameter, required)

**Response:**
```json
{
    "video_id": "string",
    "url": "string"
}
```

## Error Handling

The API handles various error cases:

- 404: Transcript not found or video unavailable
- 400: Invalid video ID or URL format
- 500: Gemini API errors or other internal errors

Each error response includes a descriptive message:

```json
{
    "detail": "Error message here"
}
```

Extracts the transcript from a YouTube video.

**Parameters:**
- `video_id` (path parameter): The YouTube video ID (e.g., "dQw4w9WgXcQ" from "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

### GET /to-url/{video_id}

Converts a YouTube video ID to its full URL.

**Parameters:**
- `video_id` (path parameter): The YouTube video ID

**Response:**
```json
{
    "video_id": "string",
    "url": "string"
}
```

### GET /to-id/

Extracts the video ID from a YouTube URL.

**Parameters:**
- `url` (query parameter): The YouTube URL (supports various formats)

**Response:**
```json
{
    "video_id": "string",
    "url": "string"
}
```

Supported URL formats:
- https://www.youtube.com/watch?v=VIDEO_ID
- https://youtu.be/VIDEO_ID
- https://www.youtube.com/embed/VIDEO_ID
- Direct video ID

**Response:**
```json
{
    "video_id": "string",
    "transcript": [
        {
            "text": "string",
            "start": number,
            "duration": number
        }
    ]
}
```

**Error Responses:**
- 404: Video not found or no transcript available
- 500: Server error

## Example Usage

Using curl:
```bash
curl http://localhost:8000/transcript/dQw4w9WgXcQ
```

Using Python requests:
```python
import requests

video_id = "dQw4w9WgXcQ"
response = requests.get(f"http://localhost:8000/transcript/{video_id}")
transcript = response.json()
```

## Error Handling

The API handles several types of errors:
- When no transcript is found for the video
- When the video is unavailable
- When the video ID is invalid
- Other potential server errors

Each error returns an appropriate HTTP status code and a descriptive error message.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Testing

The project uses pytest for testing. To run the tests:

1. Install test dependencies:
```bash
pip install -r requirements.txt
```

2. Run the tests:
```bash
pytest
```

This will:
- Run all tests in the `tests` directory
- Show test coverage information
- Display detailed test results

### Test Coverage

The tests cover:
- Successful transcript retrieval
- Handling of unavailable videos
- Handling of videos without transcripts
- Invalid video ID handling
- API response format validation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write and test your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request
