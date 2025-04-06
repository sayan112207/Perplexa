import pytest
import requests
import os
from app import get_serpapi_results, extract_content, call_gemini_api, call_mistral_api

# Modify this based on your local or server setup
#Things to Change:
#Modify BASE_URL = "http://localhost:5500" if your API runs on a different port.
BASE_URL = "http://localhost:5500"

# Mock Environment Variables
os.environ['95c27147fbae9f979cf223fa5056bac07fb569d715890e60a3ffb14679706e3d'] = 'test_serpapi_key'
os.environ['AIzaSyDIG2W0bWqDxlvA3NtVFEHBdHN_SXC4cbU'] = 'test_gemini_key'
os.environ['A1T47NIMVRMmt7aXpQDNvXxOyyaKFj99'] = 'test_mistral_key'

def test_get_serpapi_results(mocker):
    """Test SERPAPI search functionality"""
    mock_response = {"organic_results": [{"title": "Test Result", "link": "https://example.com"}]}
    mocker.patch('app.GoogleSearch.get_dict', return_value=mock_response)

    results = get_serpapi_results("test query")
    assert len(results) > 0
    assert "link" in results[0]

def test_extract_content(mocker):
    """Test web content extraction"""
    mock_html = "<html><body><p>Test content</p></body></html>"
    mocker.patch('requests.get', return_value=mocker.Mock(text=mock_html))

    content = extract_content("https://example.com")
    assert content == "Test content"

def test_call_gemini_api(mocker):
    """Test Gemini API call"""
    mock_response = {"candidates": [{"content": {"parts": [{"text": "Test Response"}]}}]}
    mocker.patch('requests.post', return_value=mocker.Mock(json=lambda: mock_response))

    response = call_gemini_api("Test Prompt")
    assert response == "Test Response"

def test_call_mistral_api(mocker):
    """Test Mistral API call"""
    mock_response = {"choices": [{"message": {"content": "Test Mistral Response"}}]}
    mocker.patch('requests.post', return_value=mocker.Mock(json=lambda: mock_response))

    response = call_mistral_api("Test Prompt")
    assert response == "Test Mistral Response"

if __name__ == "__main__":
    pytest.main()


# Run the tests
# pytest tests/test_app.py
