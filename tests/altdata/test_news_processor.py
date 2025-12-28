import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from altdata.news_processor import NewsProcessor, NewsArticle, get_news_processor


# Fixture to mock the SentimentIntensityAnalyzer
@pytest.fixture
def mock_sentiment_analyzer():
    with patch('altdata.news_processor.SentimentIntensityAnalyzer') as MockAnalyzer:
        mock_instance = MockAnalyzer.return_value
        # Default compound scores for testing
        mock_instance.polarity_scores.side_effect = lambda text: {
            'pos': 0.5, 'neg': 0.1, 'neu': 0.4, 'compound': 0.9
        } if 'positive' in text.lower() else (
            {'pos': 0.1, 'neg': 0.5, 'neu': 0.4, 'compound': -0.9} if 'negative' in text.lower() else
            {'pos': 0.3, 'neg': 0.3, 'neu': 0.4, 'compound': 0.0}
        )
        yield mock_instance

# Test initialization
def test_news_processor_init(mock_sentiment_analyzer):
    processor = NewsProcessor()
    assert processor._sentiment_analyzer is mock_sentiment_analyzer
    mock_sentiment_analyzer.polarity_scores.assert_not_called()

# Test _get_sentiment_scores
def test_get_sentiment_scores(mock_sentiment_analyzer):
    processor = NewsProcessor()
    scores = processor._get_sentiment_scores("This is a positive text.")
    assert scores['compound'] == 0.9 # Based on mock setup
    mock_sentiment_analyzer.polarity_scores.assert_called_once_with("This is a positive text.")

# Test fetch_news - basic
def test_fetch_news_basic(mock_sentiment_analyzer):
    processor = NewsProcessor()
    articles = processor.fetch_news(limit=2)
    assert len(articles) == 2
    for article in articles:
        assert 'sentiment_score' in article.to_dict()
        assert article.sentiment_score['compound'] is not None

# Test fetch_news - by symbol
def test_fetch_news_by_symbol(mock_sentiment_analyzer):
    processor = NewsProcessor()
    articles = processor.fetch_news(symbols=['AAPL'])
    assert len(articles) > 0
    for article in articles:
        assert 'AAPL' in article.symbols

# Test fetch_news - by query
def test_fetch_news_by_query(mock_sentiment_analyzer):
    processor = NewsProcessor()
    articles = processor.fetch_news(query='earnings')
    assert len(articles) > 0
    for article in articles:
        assert 'earnings' in article.headline.lower() or ('earnings' in article.summary.lower() if article.summary else False)

# Test fetch_news - by date range
def test_fetch_news_by_date_range(mock_sentiment_analyzer):
    processor = NewsProcessor()
    now = datetime.now()
    start_date = now - timedelta(hours=2)
    end_date = now - timedelta(minutes=15) # Should get 'AAPL' news
    articles = processor.fetch_news(start_date=start_date, end_date=end_date)
    
    assert len(articles) > 0
    for article in articles:
        assert start_date <= article.created_at <= end_date

# Test get_aggregated_sentiment
def test_get_aggregated_sentiment_overall(mock_sentiment_analyzer):
    processor = NewsProcessor()
    # Mock some news for aggregation
    with patch.object(processor, 'fetch_news') as mock_fetch_news:
        mock_fetch_news.return_value = [
            NewsArticle(id='1', headline='Very positive news', sentiment_score={'compound': 0.8}),
            NewsArticle(id='2', headline='Neutral report', sentiment_score={'compound': 0.1}),
            NewsArticle(id='3', headline='Negative outlook', sentiment_score={'compound': -0.7}),
        ]
        sentiment = processor.get_aggregated_sentiment()
        assert abs(sentiment['compound'] - (0.8 + 0.1 - 0.7) / 3) < 0.001
        assert sentiment['positive'] > 0
        assert sentiment['negative'] > 0

def test_get_aggregated_sentiment_no_news(mock_sentiment_analyzer):
    processor = NewsProcessor()
    with patch.object(processor, 'fetch_news') as mock_fetch_news:
        mock_fetch_news.return_value = []
        sentiment = processor.get_aggregated_sentiment()
        assert sentiment['compound'] == 0.0
        assert sentiment['neutral'] == 1.0

# Test get_news_processor singleton
def test_get_news_processor_singleton():
    processor1 = get_news_processor()
    processor2 = get_news_processor()
    assert processor1 is processor2
    assert isinstance(processor1, NewsProcessor)
