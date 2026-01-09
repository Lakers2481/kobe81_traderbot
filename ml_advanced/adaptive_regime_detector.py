"""
Adaptive Regime Detector (Stub)

This is a placeholder for future implementation.
Currently falls back to HMM regime detector.

Author: Kobe ML System
Date: 2026-01-09
Status: Stub (not yet implemented)
"""

import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class AdaptiveRegimeDetector:
    """
    Adaptive regime detector (stub implementation).

    This class exists to satisfy imports but currently just logs
    that it's not yet implemented. The system falls back to
    HMMRegimeDetector instead.
    """

    def __init__(self):
        """Initialize the adaptive regime detector stub."""
        logger.warning(
            "AdaptiveRegimeDetector is a stub - falling back to HMM detector"
        )
        self._enabled = False

    def predict(self, data: pd.DataFrame) -> Optional[str]:
        """
        Predict current market regime (stub).

        Args:
            data: Market data DataFrame

        Returns:
            None (stub implementation)
        """
        logger.debug("AdaptiveRegimeDetector.predict called (stub)")
        return None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Train the detector (stub).

        Args:
            data: Training data
        """
        logger.debug("AdaptiveRegimeDetector.fit called (stub)")
        pass
