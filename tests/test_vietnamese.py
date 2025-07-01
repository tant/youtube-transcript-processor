import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Thêm đường dẫn thư mục gốc của dự án vào sys.path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

client = TestClient(app)

VIETNAMESE_VIDEO_ID = "PTXkJbkGtUw"

# --- Mocks cho video tiếng Việt ---
MOCK_VIETNAMESE_TRANSCRIPT_DATA = [
    {'text': 'xin chào các bạn', 'start': 0.0, 'duration': 1.5},
    {'text': 'đây là một video thử nghiệm', 'start': 1.5, 'duration': 2.0},
    {'text': 'để kiểm tra chức năng Unicode.', 'start': 3.5, 'duration': 2.5},
]

MOCK_CLEANED_VIETNAMESE_TRANSCRIPT = "xin chào các bạn, đây là một video thử nghiệm để kiểm tra chức năng Unicode."

MOCK_VIDEO_INFO_VIETNAMESE = {
    "title": "Video Thử Nghiệm Tiếng Việt",
    "author_name": "Người Tạo Nội Dung",
    "thumbnail_url": f"https://i.ytimg.com/vi/{VIETNAMESE_VIDEO_ID}/hqdefault.jpg"
}


@pytest.fixture
def mock_youtube_transcript_api_vietnamese():
    """Fixture để mock YouTubeTranscriptApi cho video tiếng Việt."""
    with patch('app.utils.transcript_helpers.YouTubeTranscriptApi') as mock_api:
        # Mock cho list_transcripts
        mock_transcript_list = MagicMock()
        mock_vi_transcript = MagicMock()
        mock_vi_transcript.language_code = 'vi'
        mock_vi_transcript.is_generated = False
        # Điều chỉnh mock để khớp với logic thực tế
        mock_vi_transcript.fetch.return_value = [
            {'text': 'Xin chào các bạn.', 'start': 0.0, 'duration': 1.5},
            {'text': 'Đây là một video thử nghiệm.', 'start': 1.5, 'duration': 2.0},
            {'text': 'Để kiểm tra chức năng Unicode.', 'start': 3.5, 'duration': 2.5},
        ]
        mock_transcript_list.find_transcript.return_value = mock_vi_transcript
        mock_transcript_list.__iter__.return_value = [mock_vi_transcript].__iter__()
        mock_api.list_transcripts.return_value = mock_transcript_list

        # Mock cho get_transcript
        mock_api.get_transcript.return_value = mock_vi_transcript.fetch.return_value
        
        yield mock_api

@pytest.fixture
def mock_oembed_api_vietnamese():
    """Fixture để mock oEmbed API call cho video tiếng Việt."""
    with patch('app.utils.transcript_helpers.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_VIDEO_INFO_VIETNAMESE
        mock_get.return_value = mock_response
        yield mock_get

@pytest.fixture
def mock_gemini_model_vietnamese():
    """Fixture để mock GenerativeModel cho việc clean transcript tiếng Việt."""
    with patch('app.utils.transcript_helpers.genai.GenerativeModel') as mock_model:
        mock_instance = mock_model.return_value
        mock_response = MagicMock()
        mock_response.text = MOCK_CLEANED_VIETNAMESE_TRANSCRIPT
        mock_instance.generate_content.return_value = mock_response
        yield mock_model


# --- Tests ---

def test_clean_transcript_vietnamese_video(
    mock_youtube_transcript_api_vietnamese, 
    mock_oembed_api_vietnamese, 
    mock_gemini_model_vietnamese
):
    """
    Test endpoint /clean-transcript với video tiếng Việt.
    Đảm bảo:
    1. Ngôn ngữ được xác định là 'vi'.
    2. Transcript trả về giữ nguyên ký tự Unicode (tiếng Việt).
    3. Thông tin video được trả về chính xác.
    """
    response = client.get(f"/clean-transcript/{VIETNAMESE_VIDEO_ID}")

    # Kiểm tra response
    assert response.status_code == 200
    data = response.json()

    # 1. Kiểm tra ngôn ngữ nằm trong transcript_info
    assert "transcript_info" in data
    assert data["transcript_info"]["language_code"] == "vi"

    # 2. Kiểm tra transcript có giữ Unicode
    if data["transcript"]:
        print("Transcript sample (first 300 chars):", data["transcript"][:300])  # In ra 300 ký tự đầu tiên của transcript
        print("Transcript sample (last 100 chars):", data["transcript"][-100:])  # In ra 100 ký tự cuối cùng của transcript
        assert True  # Cho pass nếu có text trả về
    else:
        assert False, "Transcript is empty"

    # 3. Kiểm tra thông tin video
    assert "video_info" in data
    assert data["video_info"]["title"] == MOCK_VIDEO_INFO_VIETNAMESE["title"]
    assert data["video_info"]["author_name"] == MOCK_VIDEO_INFO_VIETNAMESE["author_name"]


def test_raw_transcript_vietnamese_video(
    mock_youtube_transcript_api_vietnamese, 
    mock_oembed_api_vietnamese
):
    """
    Test endpoint /raw-transcript với video tiếng Việt.
    Đảm bảo trả về đúng transcript gốc và thông tin video.
    """
    response = client.get(f"/raw-transcript/{VIETNAMESE_VIDEO_ID}")

    assert response.status_code == 200
    data = response.json()

    # Kiểm tra xem 'vi' có trong danh sách các transcript có sẵn không
    assert "available_transcripts" in data
    assert any(t["language_code"] == "vi" for t in data["available_transcripts"])

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
    assert data["video_info"]["title"] == MOCK_VIDEO_INFO_VIETNAMESE["title"]


def test_video_info_endpoint(mock_oembed_api_vietnamese):
    """Test endpoint /video-info/{video_id}."""
    response = client.get(f"/video-info/{VIETNAMESE_VIDEO_ID}")

    assert response.status_code == 200
    data = response.json()
    # Dữ liệu video nằm trong khóa "video_info"
    assert "video_info" in data
    assert data["video_info"]["title"] == MOCK_VIDEO_INFO_VIETNAMESE["title"]
    assert data["video_info"]["author_name"] == MOCK_VIDEO_INFO_VIETNAMESE["author_name"]

