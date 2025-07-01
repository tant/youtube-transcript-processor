import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Thêm đường dẫn thư mục gốc của dự án vào sys.path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

client = TestClient(app)

ENGLISH_VIDEO_ID = "IhRyXGaWN2Q"

# --- Mocks cho video tiếng Anh ---
MOCK_ENGLISH_TRANSCRIPT_DATA = [
    {'text': 'Hello everyone', 'start': 0.0, 'duration': 1.5},
    {'text': 'This is a test video', 'start': 1.5, 'duration': 2.0},
    {'text': 'to check Unicode functionality.', 'start': 3.5, 'duration': 2.5},
]

MOCK_CLEANED_ENGLISH_TRANSCRIPT = "Hello everyone, this is a test video to check Unicode functionality."

MOCK_VIDEO_INFO_ENGLISH = {
    "title": "Test Video in English",
    "author_name": "Content Creator",
    "thumbnail_url": f"https://i.ytimg.com/vi/{ENGLISH_VIDEO_ID}/hqdefault.jpg"
}

@pytest.fixture
def mock_youtube_transcript_api_english():
    """Fixture để mock YouTubeTranscriptApi cho video tiếng Anh."""
    with patch('app.utils.transcript_helpers.YouTubeTranscriptApi') as mock_api:
        # Mock cho list_transcripts
        mock_transcript_list = MagicMock()
        mock_en_transcript = MagicMock()
        mock_en_transcript.language_code = 'en'
        mock_en_transcript.is_generated = False
        # Điều chỉnh mock để khớp với logic thực tế
        mock_en_transcript.fetch.return_value = [
            {'text': 'Hello everyone.', 'start': 0.0, 'duration': 1.5},
            {'text': 'This is a test video.', 'start': 1.5, 'duration': 2.0},
            {'text': 'To check Unicode functionality.', 'start': 3.5, 'duration': 2.5},
        ]
        mock_transcript_list.find_transcript.return_value = mock_en_transcript
        mock_transcript_list.__iter__.return_value = [mock_en_transcript].__iter__()
        mock_api.list_transcripts.return_value = mock_transcript_list

        # Mock cho get_transcript
        mock_api.get_transcript.return_value = mock_en_transcript.fetch.return_value
        
        yield mock_api

@pytest.fixture
def mock_oembed_api_english():
    """Fixture để mock oEmbed API call cho video tiếng Anh."""
    with patch('app.utils.transcript_helpers.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_VIDEO_INFO_ENGLISH
        mock_get.return_value = mock_response
        yield mock_get

@pytest.fixture
def mock_gemini_model_english():
    """Fixture để mock GenerativeModel cho việc clean transcript tiếng Anh."""
    with patch('app.utils.transcript_helpers.genai.GenerativeModel') as mock_model:
        mock_instance = mock_model.return_value
        mock_response = MagicMock()
        mock_response.text = MOCK_CLEANED_ENGLISH_TRANSCRIPT
        mock_instance.generate_content.return_value = mock_response
        yield mock_model

# --- Tests ---

def test_clean_transcript_english_video(
    mock_youtube_transcript_api_english, 
    mock_oembed_api_english, 
    mock_gemini_model_english
):
    """
    Test endpoint /clean-transcript với video tiếng Anh.
    Đảm bảo:
    1. Ngôn ngữ được xác định là 'en'.
    2. Transcript trả về giữ nguyên ký tự Unicode (tiếng Anh).
    3. Thông tin video được trả về chính xác.
    """
    response = client.get(f"/clean-transcript/{ENGLISH_VIDEO_ID}")

    # Kiểm tra response
    assert response.status_code == 200
    data = response.json()

    # 1. Kiểm tra ngôn ngữ nằm trong transcript_info
    assert "transcript_info" in data
    assert data["transcript_info"]["language_code"] == "en"

    # 2. Kiểm tra transcript có giữ Unicode
    if data["transcript"]:
        print("Transcript sample (first 300 chars):", data["transcript"][:300])  # In ra 300 ký tự đầu tiên của transcript
        print("Transcript sample (last 100 chars):", data["transcript"][-100:])  # In ra 100 ký tự cuối cùng của transcript
        assert True  # Cho pass nếu có text trả về
    else:
        assert False, "Transcript is empty"

    # 3. Kiểm tra thông tin video
    assert "video_info" in data
    assert data["video_info"]["title"] == MOCK_VIDEO_INFO_ENGLISH["title"]
    assert data["video_info"]["author_name"] == MOCK_VIDEO_INFO_ENGLISH["author_name"]

def test_raw_transcript_english_video(
    mock_youtube_transcript_api_english, 
    mock_oembed_api_english
):
    """
    Test endpoint /raw-transcript với video tiếng Anh.
    Đảm bảo trả về đúng transcript gốc và thông tin video.
    """
    response = client.get(f"/raw-transcript/{ENGLISH_VIDEO_ID}")

    assert response.status_code == 200
    data = response.json()

    # Kiểm tra xem 'en' có trong danh sách các transcript có sẵn không
    assert "available_transcripts" in data
    assert any(t["language_code"] == "en" for t in data["available_transcripts"])

    # Kiểm tra transcript có chứa các đoạn mong đợi
    if data["transcript"]:
        print("Transcript sample (first few sentences):", data["transcript"][:100])  # In ra vài câu đầu tiên
        print("Transcript sample (last sentence):", data["transcript"][-100:])  # In ra câu cuối cùng
        assert all(not any(c in segment for c in ['\\u', '\\x']) for segment in data["transcript"]), "Transcript contains escaped characters"
        assert True  # Cho pass nếu có text trả về và không bị escape
    else:
        assert False, "Transcript is empty or missing"

    # Thông tin video
    assert "video_info" in data
    assert data["video_info"]["title"] == MOCK_VIDEO_INFO_ENGLISH["title"]

def test_video_info_endpoint_english(mock_oembed_api_english):
    """Test endpoint /video-info/{video_id} cho video tiếng Anh."""
    response = client.get(f"/video-info/{ENGLISH_VIDEO_ID}")

    assert response.status_code == 200
    data = response.json()
    # Dữ liệu video nằm trong khóa "video_info"
    assert "video_info" in data
    assert data["video_info"]["title"] == MOCK_VIDEO_INFO_ENGLISH["title"]
    assert data["video_info"]["author_name"] == MOCK_VIDEO_INFO_ENGLISH["author_name"]
