import pytest
from fastapi.testclient import TestClient
from youtube_transcript_api import NoTranscriptFound, VideoUnavailable
from unittest.mock import patch
from app.main import app

client = TestClient(app)

# Test data
VALID_VIDEO_ID = "dQw4w9WgXcQ"
INVALID_VIDEO_ID = "invalid_id"
VALID_YOUTUBE_URLS = [
    f"https://www.youtube.com/watch?v={VALID_VIDEO_ID}",
    f"https://youtu.be/{VALID_VIDEO_ID}",
    f"https://www.youtube.com/embed/{VALID_VIDEO_ID}",
    f"https://youtube.com/watch?v={VALID_VIDEO_ID}&feature=share",
    VALID_VIDEO_ID  # Just the ID should also work
]
INVALID_YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=invalid",
    "https://youtube.com/invalid",
    "https://youtube.com/watch?v=",
    "https://youtube.com/watch?v=123"  # quá ngắn
]
MOCK_TRANSCRIPT = [
    {
        "text": "Never gonna give you up",
        "start": 0.0,
        "duration": 4.0
    },
    {
        "text": "Never gonna let you down",
        "start": 4.0,
        "duration": 4.0
    }
]

def test_read_root():
    """Test the root endpoint returns 404 as it's not defined"""
    response = client.get("/")
    assert response.status_code == 404

@patch('youtube_transcript_api.YouTubeTranscriptApi.get_transcript')
def test_get_transcript_success(mock_get_transcript):
    """Test successful transcript retrieval"""
    mock_get_transcript.return_value = MOCK_TRANSCRIPT
    
    response = client.get(f"/transcript/{VALID_VIDEO_ID}")
    
    assert response.status_code == 200
    assert response.json() == {
        "video_id": VALID_VIDEO_ID,
        "transcript": ["Never gonna give you up", "Never gonna let you down"]
    }
    mock_get_transcript.assert_called_once_with(VALID_VIDEO_ID)

@patch('youtube_transcript_api.YouTubeTranscriptApi.get_transcript')
def test_get_transcript_no_transcript(mock_get_transcript):
    """Test when no transcript is available"""
    mock_get_transcript.side_effect = NoTranscriptFound(VALID_VIDEO_ID, ['en'], None)
    
    response = client.get(f"/transcript/{VALID_VIDEO_ID}")
    
    assert response.status_code == 404
    assert response.json() == {"detail": "No transcript found for this video"}

@patch('youtube_transcript_api.YouTubeTranscriptApi.get_transcript')
def test_get_transcript_video_unavailable(mock_get_transcript):
    """Test when video is unavailable"""
    mock_get_transcript.side_effect = VideoUnavailable(VALID_VIDEO_ID)
    
    response = client.get(f"/transcript/{VALID_VIDEO_ID}")
    
    assert response.status_code == 404
    assert response.json() == {"detail": "Video is unavailable"}

def test_get_transcript_invalid_video_id():
    """Test with invalid video ID format"""
    response = client.get(f"/transcript/{INVALID_VIDEO_ID}")
    assert response.status_code == 404

def test_get_full_url_success():
    """Test successful video ID to URL conversion"""
    response = client.get(f"/to-url/{VALID_VIDEO_ID}")
    
    assert response.status_code == 200
    assert response.json() == {
        "video_id": VALID_VIDEO_ID,
        "url": f"https://www.youtube.com/watch?v={VALID_VIDEO_ID}"
    }

def test_get_full_url_invalid_id():
    """Test URL conversion with invalid video ID"""
    response = client.get(f"/to-url/{INVALID_VIDEO_ID}")
    
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid video ID format"}

def test_get_video_id_success():
    """Test successful URL to video ID extraction for various URL formats"""
    for url in VALID_YOUTUBE_URLS:
        response = client.get("/to-id/", params={"url": url})
        
        assert response.status_code == 200
        assert response.json() == {
            "video_id": VALID_VIDEO_ID,
            "url": url
        }

def test_get_video_id_invalid_url():
    """Test video ID extraction with invalid URLs"""
    for url in INVALID_YOUTUBE_URLS:
        print(f"\nTesting invalid URL: {url!r}")
        response = client.get("/to-id/", params={"url": url})
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")
        
        assert response.status_code == 400
        assert response.json() == {"detail": "Could not extract video ID from URL"}

@patch('google.generative.ai.GenerativeModel.generate_content')
@patch('youtube_transcript_api.YouTubeTranscriptApi.get_transcript')
def test_get_transcript_summary_success(mock_get_transcript, mock_generate_content):
    """Test successful transcript summary retrieval"""
    # Mock transcript response
    mock_get_transcript.return_value = MOCK_TRANSCRIPT
    
    # Mock Gemini response
    class MockGeminiResponse:
        text = "Never gonna give you up. Never gonna let you down."
    mock_generate_content.return_value = MockGeminiResponse()
    
    response = client.get(f"/transcript-summary/{VALID_VIDEO_ID}")
    
    assert response.status_code == 200
    assert response.json() == {
        "video_id": VALID_VIDEO_ID,
        "transcript": "Never gonna give you up. Never gonna let you down."
    }
    mock_get_transcript.assert_called_once_with(VALID_VIDEO_ID)
    mock_generate_content.assert_called_once()
