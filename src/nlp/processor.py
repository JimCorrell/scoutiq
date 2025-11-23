"""
NLP Pipeline for processing baseball scouting reports and extracting insights
"""
import re
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TextPreprocessor:
    """
    Preprocess and clean text data
    """
    
    def __init__(self):
        """Initialize text preprocessor"""
        logger.info("Initialized TextPreprocessor")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep baseball-specific notation
        # Keep: hyphens, slashes, numbers, grades (e.g., "60-grade")
        text = re.sub(r'[^\w\s\-/,.]', '', text)
        
        return text.strip()
    
    def split_reports(self, combined_text: str, separator: str = '[REPORT_SEP]') -> List[str]:
        """
        Split combined reports into individual reports
        
        Args:
            combined_text: Text with multiple reports
            separator: Separator string
            
        Returns:
            List of individual reports
        """
        if separator in combined_text:
            return [r.strip() for r in combined_text.split(separator) if r.strip()]
        return [combined_text]
    
    def normalize_grades(self, text: str) -> str:
        """
        Normalize tool grade notation (e.g., "plus-plus" -> "70", "above-average" -> "55")
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized grades
        """
        grade_mapping = {
            r'\b80-grade\b': '80',
            r'\bplus-plus\b': '70',
            r\b70-grade\b': '70',
            r'\bplus\b(?!-plus)': '60',
            r\b60-grade\b': '60',
            r'\babove-average\b': '55',
            r\b55-grade\b': '55',
            r'\baverage\b': '50',
            r\b50-grade\b': '50',
            r'\bbelow-average\b': '45',
            r\b45-grade\b': '45',
            r'\bfringe\b': '40',
            r\b40-grade\b': '40',
            r'\bpoor\b': '30',
            r\b30-grade\b': '30',
        }
        
        normalized_text = text
        for pattern, replacement in grade_mapping.items():
            normalized_text = re.sub(pattern, f"GRADE_{replacement}", normalized_text, flags=re.IGNORECASE)
        
        return normalized_text


class ToolGradeExtractor:
    """
    Extract baseball tool grades from scouting reports
    Tools: Hit, Power, Run, Arm, Field (for position players)
           Fastball, Curveball, Slider, Changeup, Control/Command (for pitchers)
    """
    
    # 20-80 scouting scale
    GRADE_PATTERNS = {
        '20': r'\b20[-\s]?grade\b',
        '30': r'\b30[-\s]?grade\b|\bpoor\b',
        '40': r'\b40[-\s]?grade\b|\bfringe\b|\bbelow[-\s]?average\b',
        '45': r'\b45[-\s]?grade\b',
        '50': r'\b50[-\s]?grade\b|\baverage\b',
        '55': r'\b55[-\s]?grade\b|\babove[-\s]?average\b',
        '60': r'\b60[-\s]?grade\b|\bplus\b(?![-\s]?plus)',
        '70': r'\b70[-\s]?grade\b|\bplus[-\s]?plus\b',
        '80': r'\b80[-\s]?grade\b',
    }
    
    POSITION_TOOLS = ['hit', 'power', 'run', 'arm', 'field', 'speed']
    PITCHER_TOOLS = ['fastball', 'curveball', 'slider', 'changeup', 'control', 'command']
    
    def __init__(self):
        """Initialize tool grade extractor"""
        logger.info("Initialized ToolGradeExtractor")
    
    def extract_grades(self, text: str) -> Dict[str, Optional[int]]:
        """
        Extract tool grades from text
        
        Args:
            text: Scouting report text
            
        Returns:
            Dictionary of tool: grade pairs
        """
        text_lower = text.lower()
        grades = {}
        
        # Check for position player tools
        for tool in self.POSITION_TOOLS:
            grade = self._find_grade_for_tool(text_lower, tool)
            if grade:
                grades[f'{tool}_grade'] = grade
        
        # Check for pitcher tools
        for tool in self.PITCHER_TOOLS:
            grade = self._find_grade_for_tool(text_lower, tool)
            if grade:
                grades[f'{tool}_grade'] = grade
        
        return grades
    
    def _find_grade_for_tool(self, text: str, tool: str) -> Optional[int]:
        """
        Find grade for specific tool in text
        
        Args:
            text: Scouting report text (lowercase)
            tool: Tool name
            
        Returns:
            Grade value or None
        """
        # Look for patterns like "plus power" or "power: 60-grade"
        tool_pattern = rf'\b{tool}\b.{{0,30}}'
        
        matches = re.finditer(tool_pattern, text)
        
        for match in matches:
            context = text[match.start():match.end() + 50]
            
            for grade_value, grade_pattern in self.GRADE_PATTERNS.items():
                if re.search(grade_pattern, context, re.IGNORECASE):
                    return int(grade_value)
        
        return None
    
    def extract_overall_grade(self, text: str) -> Optional[int]:
        """
        Extract overall prospect grade (OFP - Overall Future Potential)
        
        Args:
            text: Scouting report text
            
        Returns:
            Overall grade or None
        """
        ofp_pattern = r'(?:ofp|overall|future\s+potential).{0,20}'
        
        matches = re.finditer(ofp_pattern, text.lower())
        
        for match in matches:
            context = text[match.start():match.end() + 30]
            
            for grade_value, grade_pattern in self.GRADE_PATTERNS.items():
                if re.search(grade_pattern, context, re.IGNORECASE):
                    return int(grade_value)
        
        return None


class SentimentAnalyzer:
    """
    Analyze sentiment of scouting reports
    """
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        if not TEXTBLOB_AVAILABLE:
            logger.warning("TextBlob not available. Sentiment analysis will be limited.")
        logger.info("Initialized SentimentAnalyzer")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        if not TEXTBLOB_AVAILABLE:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def classify_sentiment(self, polarity: float) -> str:
        """
        Classify sentiment as positive, negative, or neutral
        
        Args:
            polarity: Sentiment polarity score
            
        Returns:
            Sentiment classification
        """
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'


class KeywordExtractor:
    """
    Extract baseball-specific keywords and phrases
    """
    
    STRENGTH_KEYWORDS = [
        'plus', 'excellent', 'strong', 'good', 'solid', 'advanced',
        'impressive', 'potential', 'upside', 'tools', 'elite'
    ]
    
    CONCERN_KEYWORDS = [
        'concern', 'weakness', 'issue', 'struggle', 'limited', 'needs improvement',
        'injury', 'inconsistent', 'question', 'risk', 'lacks'
    ]
    
    POSITION_KEYWORDS = [
        'catcher', 'first base', 'second base', 'third base', 'shortstop',
        'outfield', 'left field', 'center field', 'right field',
        'pitcher', 'starter', 'reliever', 'closer'
    ]
    
    def __init__(self):
        """Initialize keyword extractor"""
        logger.info("Initialized KeywordExtractor")
    
    def extract_keywords(self, text: str) -> Dict[str, int]:
        """
        Extract and count keywords
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of keyword counts
        """
        text_lower = text.lower()
        
        return {
            'strength_mentions': sum(1 for kw in self.STRENGTH_KEYWORDS if kw in text_lower),
            'concern_mentions': sum(1 for kw in self.CONCERN_KEYWORDS if kw in text_lower),
            'position_mentions': sum(1 for kw in self.POSITION_KEYWORDS if kw in text_lower)
        }
    
    def extract_skills(self, text: str) -> Dict[str, bool]:
        """
        Extract mentions of specific skills
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of skill mentions
        """
        text_lower = text.lower()
        
        skills = {
            'power': any(word in text_lower for word in ['power', 'home run', 'extra base']),
            'speed': any(word in text_lower for word in ['speed', 'fast', 'quick', 'steal']),
            'contact': any(word in text_lower for word in ['contact', 'hit tool', 'bat-to-ball']),
            'defense': any(word in text_lower for word in ['defense', 'glove', 'fielding']),
            'arm': any(word in text_lower for word in ['arm', 'throwing', 'velocity']),
            'plate_discipline': any(word in text_lower for word in ['discipline', 'patience', 'walks'])
        }
        
        return skills


class NLPPipeline:
    """
    Complete NLP pipeline for processing scouting reports
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize NLP pipeline
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP
        """
        self.preprocessor = TextPreprocessor()
        self.grade_extractor = ToolGradeExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        
        self.nlp = None
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_lg')
                logger.info("Loaded spaCy model: en_core_web_lg")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {str(e)}")
        
        logger.info("Initialized NLPPipeline")
    
    def process_report(self, text: str) -> Dict:
        """
        Process a single scouting report
        
        Args:
            text: Scouting report text
            
        Returns:
            Dictionary of extracted features
        """
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Extract tool grades
        grades = self.grade_extractor.extract_grades(cleaned_text)
        overall_grade = self.grade_extractor.extract_overall_grade(cleaned_text)
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.analyze_sentiment(cleaned_text)
        sentiment_class = self.sentiment_analyzer.classify_sentiment(sentiment['polarity'])
        
        # Keyword extraction
        keywords = self.keyword_extractor.extract_keywords(cleaned_text)
        skills = self.keyword_extractor.extract_skills(cleaned_text)
        
        # Combine features
        features = {
            'cleaned_text': cleaned_text,
            'text_length': len(cleaned_text),
            'word_count': len(cleaned_text.split()),
            'overall_grade': overall_grade,
            'sentiment_polarity': sentiment['polarity'],
            'sentiment_subjectivity': sentiment['subjectivity'],
            'sentiment_class': sentiment_class,
            **grades,
            **keywords,
            **{f'skill_{k}': int(v) for k, v in skills.items()}
        }
        
        return features
    
    def process_dataframe(self, df: pd.DataFrame, text_col: str = 'report_text') -> pd.DataFrame:
        """
        Process all reports in a DataFrame
        
        Args:
            df: DataFrame with scouting reports
            text_col: Column containing report text
            
        Returns:
            DataFrame with extracted NLP features
        """
        logger.info(f"Processing {len(df)} reports")
        
        # Process each report
        features_list = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} reports")
            
            try:
                features = self.process_report(row[text_col])
                features['player_id'] = row.get('player_id', None)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Error processing report at index {idx}: {str(e)}")
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        logger.info(f"Extracted {len(features_df.columns)} NLP features")
        return features_df
