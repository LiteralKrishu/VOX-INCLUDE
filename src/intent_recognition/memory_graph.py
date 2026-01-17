"""Conversational memory graph for context tracking."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import deque
import json


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    timestamp: datetime
    text: str
    speaker: str  # "user" or "system"
    intent: Optional[str] = None
    emotion: Optional[str] = None
    confidence: float = 0.0
    entities: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "text": self.text,
            "speaker": self.speaker,
            "intent": self.intent,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "entities": self.entities
        }


@dataclass
class TopicNode:
    """A topic in the conversation graph."""
    name: str
    first_mentioned: datetime
    last_mentioned: datetime
    mention_count: int = 1
    related_topics: List[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 to 1


class MemoryGraph:
    """Conversational memory graph for tracking context and patterns.
    
    Maintains:
    - Short-term memory: Recent conversation turns
    - Topic tracking: Topics discussed and their relationships
    - Pattern detection: Repeated questions, misunderstandings
    - Context window: Sliding window for temporal relevance
    """
    
    def __init__(
        self,
        max_turns: int = 20,
        context_window: int = 5,
        topic_decay_factor: float = 0.9
    ):
        """Initialize memory graph.
        
        Args:
            max_turns: Maximum conversation turns to store
            context_window: Number of recent turns for immediate context
            topic_decay_factor: Decay factor for topic relevance over time
        """
        self.max_turns = max_turns
        self.context_window = context_window
        self.topic_decay_factor = topic_decay_factor
        
        # Conversation history
        self._turns: deque = deque(maxlen=max_turns)
        
        # Topic tracking
        self._topics: Dict[str, TopicNode] = {}
        
        # Pattern tracking
        self._repeated_questions: List[str] = []
        self._clarification_count: int = 0
        self._misunderstanding_indicators: int = 0
        
        # Session metadata
        self._session_start: datetime = datetime.now()
        self._total_turns: int = 0
    
    def add_turn(
        self,
        text: str,
        speaker: str = "user",
        intent: Optional[str] = None,
        emotion: Optional[str] = None,
        confidence: float = 0.0,
        entities: Optional[Dict[str, str]] = None
    ) -> ConversationTurn:
        """Add a conversation turn to memory.
        
        Args:
            text: The spoken/written text
            speaker: "user" or "system"
            intent: Classified intent
            emotion: Detected emotion
            confidence: Classification confidence
            entities: Extracted entities
            
        Returns:
            The created ConversationTurn
        """
        turn = ConversationTurn(
            timestamp=datetime.now(),
            text=text,
            speaker=speaker,
            intent=intent,
            emotion=emotion,
            confidence=confidence,
            entities=entities or {}
        )
        
        self._turns.append(turn)
        self._total_turns += 1
        
        # Track patterns
        self._track_patterns(turn)
        
        # Extract and track topics (simple keyword extraction)
        self._extract_topics(turn)
        
        return turn
    
    def _track_patterns(self, turn: ConversationTurn) -> None:
        """Track conversation patterns for cognitive state estimation."""
        # Track clarification requests
        if turn.intent == "clarification":
            self._clarification_count += 1
            self._misunderstanding_indicators += 1
        
        # Track repeated questions
        if turn.intent == "question" and turn.speaker == "user":
            recent_questions = [
                t.text.lower() for t in self._turns
                if t.intent == "question" and t.speaker == "user"
            ]
            if self._is_similar_question(turn.text, recent_questions):
                self._repeated_questions.append(turn.text)
                self._misunderstanding_indicators += 1
    
    def _is_similar_question(self, text: str, previous: List[str]) -> bool:
        """Check if a question is similar to previous ones."""
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        for prev in previous[-5:]:  # Check last 5 questions
            prev_words = set(prev.split())
            # Simple Jaccard similarity
            intersection = len(text_words & prev_words)
            union = len(text_words | prev_words)
            if union > 0 and intersection / union > 0.6:
                return True
        return False
    
    def _extract_topics(self, turn: ConversationTurn) -> None:
        """Extract and track topics from text (simple keyword extraction)."""
        # Simple noun extraction (in production, use NER or keyword extraction)
        words = turn.text.lower().split()
        # Filter common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "i", "you", "we", "they", "it", "this", "that", "and", "or",
                     "but", "if", "then", "so", "to", "of", "in", "on", "at"}
        
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        
        now = datetime.now()
        for keyword in keywords[:5]:  # Limit to 5 keywords per turn
            if keyword in self._topics:
                self._topics[keyword].mention_count += 1
                self._topics[keyword].last_mentioned = now
            else:
                self._topics[keyword] = TopicNode(
                    name=keyword,
                    first_mentioned=now,
                    last_mentioned=now
                )
    
    def get_context(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """Get recent conversation context.
        
        Args:
            n: Number of turns to retrieve (default: context_window)
            
        Returns:
            List of recent conversation turns
        """
        n = n or self.context_window
        return list(self._turns)[-n:]
    
    def get_context_text(self, n: Optional[int] = None) -> str:
        """Get context as formatted text."""
        context = self.get_context(n)
        return "\n".join([
            f"{turn.speaker}: {turn.text}"
            for turn in context
        ])
    
    def get_active_topics(self, top_n: int = 5) -> List[TopicNode]:
        """Get most active/relevant topics."""
        if not self._topics:
            return []
        
        # Score topics by recency and frequency
        now = datetime.now()
        scored = []
        for topic in self._topics.values():
            age_seconds = (now - topic.last_mentioned).total_seconds()
            decay = self.topic_decay_factor ** (age_seconds / 60)  # Decay per minute
            score = topic.mention_count * decay
            scored.append((score, topic))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in scored[:top_n]]
    
    def get_conversation_metrics(self) -> Dict[str, Any]:
        """Get metrics about the conversation for cognitive state estimation."""
        if not self._turns:
            return {
                "total_turns": 0,
                "session_duration_seconds": 0,
                "clarification_rate": 0.0,
                "misunderstanding_score": 0,
                "avg_turn_length": 0,
                "question_rate": 0.0
            }
        
        session_duration = (datetime.now() - self._session_start).total_seconds()
        
        user_turns = [t for t in self._turns if t.speaker == "user"]
        question_count = sum(1 for t in user_turns if t.intent == "question")
        
        avg_length = sum(len(t.text.split()) for t in self._turns) / len(self._turns)
        
        return {
            "total_turns": self._total_turns,
            "session_duration_seconds": session_duration,
            "clarification_rate": self._clarification_count / max(self._total_turns, 1),
            "misunderstanding_score": self._misunderstanding_indicators,
            "avg_turn_length": avg_length,
            "question_rate": question_count / max(len(user_turns), 1) if user_turns else 0,
            "repeated_questions": len(self._repeated_questions),
            "active_topics": len(self.get_active_topics())
        }
    
    def clear(self) -> None:
        """Clear conversation history."""
        self._turns.clear()
        self._topics.clear()
        self._repeated_questions.clear()
        self._clarification_count = 0
        self._misunderstanding_indicators = 0
        self._session_start = datetime.now()
        self._total_turns = 0
    
    def to_dict(self) -> Dict:
        """Export memory graph to dictionary."""
        return {
            "turns": [t.to_dict() for t in self._turns],
            "topics": {k: {
                "name": v.name,
                "mention_count": v.mention_count,
                "first_mentioned": v.first_mentioned.isoformat(),
                "last_mentioned": v.last_mentioned.isoformat()
            } for k, v in self._topics.items()},
            "metrics": self.get_conversation_metrics()
        }
