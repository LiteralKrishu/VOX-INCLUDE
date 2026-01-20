"""
VOX-INCLUDE: Consent Management Module

Implements granular consent management:
- Multiple permission levels
- Right to opaqueness
- Audit logging
- Data export and deletion
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class PermissionLevel(Enum):
    """User consent levels for data processing."""
    NONE = auto()           # No processing allowed
    VOICE_ONLY = auto()     # Speech emotion + intent only
    VOICE_BEHAVIORAL = auto()  # + interaction patterns
    FULL_MULTIMODAL = auto()   # + optional facial/visual analysis


class DataCategory(Enum):
    """Categories of data that can be processed."""
    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    EMOTION = "emotion"
    INTENT = "intent"
    BEHAVIORAL = "behavioral"
    FACIAL = "facial"
    AGGREGATE = "aggregate"


@dataclass
class ConsentRecord:
    """Records a consent decision."""
    timestamp: datetime
    permission_level: PermissionLevel
    categories_allowed: List[DataCategory]
    user_id: str
    session_id: str
    explicit_consent: bool = True
    
    
@dataclass
class AuditEntry:
    """Audit log entry for data access."""
    timestamp: datetime
    action: str
    data_category: DataCategory
    purpose: str
    user_id: str
    session_id: str
    

class ConsentManager:
    """
    Manages user consent for data processing with granular controls.
    """

    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.current_level = PermissionLevel.NONE
        self.consent_history: List[ConsentRecord] = []
        self.audit_log: List[AuditEntry] = []
        self._opaque_mode = False
        
    def set_permission_level(self, level: PermissionLevel) -> None:
        """
        Update user's consent level.
        """
        record = ConsentRecord(
            timestamp=datetime.now(),
            permission_level=level,
            categories_allowed=self._get_allowed_categories(level),
            user_id=self.user_id,
            session_id=self.session_id,
        )
        self.consent_history.append(record)
        self.current_level = level
        self._log_audit("CONSENT_UPDATE", DataCategory.AGGREGATE, f"Changed to {level.name}")

    def _get_allowed_categories(self, level: PermissionLevel) -> List[DataCategory]:
        """Map permission level to allowed data categories."""
        if level == PermissionLevel.NONE:
            return []
        elif level == PermissionLevel.VOICE_ONLY:
            return [DataCategory.AUDIO, DataCategory.TRANSCRIPT, 
                    DataCategory.EMOTION, DataCategory.INTENT]
        elif level == PermissionLevel.VOICE_BEHAVIORAL:
            return [DataCategory.AUDIO, DataCategory.TRANSCRIPT,
                    DataCategory.EMOTION, DataCategory.INTENT,
                    DataCategory.BEHAVIORAL]
        elif level == PermissionLevel.FULL_MULTIMODAL:
            return list(DataCategory)
        return []

    def is_allowed(self, category: DataCategory) -> bool:
        """
        Check if processing a data category is allowed under current consent.
        """
        if self.current_level == PermissionLevel.NONE:
            return False
        allowed = self._get_allowed_categories(self.current_level)
        return category in allowed

    def enable_opaque_mode(self) -> None:
        """
        Enable 'Right to Opaqueness' - user receives benefits 
        but system doesn't retain/analyze detailed data.
        """
        self._opaque_mode = True
        self._log_audit("OPAQUE_MODE_ENABLED", DataCategory.AGGREGATE, 
                       "User enabled opaque mode")

    def disable_opaque_mode(self) -> None:
        """Disable opaque mode."""
        self._opaque_mode = False
        self._log_audit("OPAQUE_MODE_DISABLED", DataCategory.AGGREGATE,
                       "User disabled opaque mode")

    def is_opaque(self) -> bool:
        """Check if opaque mode is active."""
        return self._opaque_mode

    def _log_audit(self, action: str, category: DataCategory, purpose: str) -> None:
        """Add entry to audit log."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            action=action,
            data_category=category,
            purpose=purpose,
            user_id=self.user_id,
            session_id=self.session_id,
        )
        self.audit_log.append(entry)

    def log_data_access(self, category: DataCategory, purpose: str) -> bool:
        """
        Log a data access attempt. Returns True if access is allowed.
        """
        allowed = self.is_allowed(category)
        action = "DATA_ACCESS_ALLOWED" if allowed else "DATA_ACCESS_DENIED"
        self._log_audit(action, category, purpose)
        return allowed

    def export_user_data(self) -> Dict[str, Any]:
        """
        Export all data associated with user (GDPR compliance).
        """
        self._log_audit("DATA_EXPORT", DataCategory.AGGREGATE, "User requested data export")
        
        return {
            "user_id": self.user_id,
            "current_permission_level": self.current_level.name,
            "opaque_mode": self._opaque_mode,
            "consent_history": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "level": r.permission_level.name,
                    "categories": [c.value for c in r.categories_allowed],
                }
                for r in self.consent_history
            ],
            "audit_log": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "action": e.action,
                    "category": e.data_category.value,
                    "purpose": e.purpose,
                }
                for e in self.audit_log
            ],
        }

    def request_deletion(self) -> Dict[str, Any]:
        """
        Process 'Right to be Forgotten' request.
        Returns confirmation of what was deleted.
        """
        self._log_audit("DELETION_REQUEST", DataCategory.AGGREGATE, 
                       "User requested data deletion")
        
        # Record what's being deleted before clearing
        deleted_info = {
            "consent_records_deleted": len(self.consent_history),
            "audit_entries_deleted": len(self.audit_log),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "deletion_timestamp": datetime.now().isoformat(),
        }
        
        # Clear all user data
        self.consent_history.clear()
        self.audit_log.clear()
        self.current_level = PermissionLevel.NONE
        self._opaque_mode = False
        
        return deleted_info


class TransparencyDashboard:
    """
    Provides transparency about AI decisions and data usage.
    """

    @staticmethod
    def explain_decision(
        emotion_result: Dict[str, Any],
        cognitive_result: Dict[str, Any],
        intervention_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate human-readable explanation of AI decision.
        
        Implements 'Explainable AI Layer' requirement.
        """
        explanation = {
            "summary": "",
            "factors": [],
            "confidence_breakdown": {},
            "data_used": [],
        }
        
        # Emotion explanation
        if emotion_result:
            emotion_label = emotion_result.get("emotion", "unknown")
            confidence = emotion_result.get("confidence", 0)
            explanation["summary"] = f"Detected '{emotion_label}' with {confidence:.0%} confidence."
            explanation["confidence_breakdown"]["emotion"] = confidence
            explanation["factors"].append({
                "factor": "Voice Tone Analysis",
                "contribution": "Primary signal for emotion detection",
                "weight": 0.6,
            })
            explanation["data_used"].append("audio_features")

        # Cognitive state explanation
        if cognitive_result:
            state = cognitive_result.get("state", "unknown")
            cog_conf = cognitive_result.get("confidence", 0)
            explanation["summary"] += f" Cognitive state: '{state}'."
            explanation["confidence_breakdown"]["cognitive_state"] = cog_conf
            explanation["factors"].append({
                "factor": "Speech Pattern Analysis",
                "contribution": "Pace, pauses, and hesitations analyzed",
                "weight": 0.3,
            })
            explanation["data_used"].append("behavioral_patterns")

        # Intervention explanation
        if intervention_result and intervention_result.get("should_intervene"):
            intervention_type = intervention_result.get("intervention_type", "none")
            explanation["summary"] += f" Intervention recommendation: {intervention_type}."
            explanation["factors"].append({
                "factor": "Combined Analysis Rule",
                "contribution": f"Triggered by detected state pattern",
                "weight": 0.1,
            })

        return explanation

    @staticmethod
    def get_processing_summary(consent_manager: ConsentManager) -> Dict[str, Any]:
        """
        Get summary of what data is being processed and why.
        """
        return {
            "permission_level": consent_manager.current_level.name,
            "opaque_mode": consent_manager.is_opaque(),
            "data_categories_active": [
                c.value for c in consent_manager._get_allowed_categories(
                    consent_manager.current_level
                )
            ],
            "recent_access_count": len(consent_manager.audit_log),
            "session_id": consent_manager.session_id,
        }
