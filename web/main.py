"""
Web Dashboard - FastAPI Application
=====================================

This module provides a FastAPI web application to serve as a monitoring and
introspection dashboard for the trading bot. It exposes various API endpoints
to query the real-time status, performance metrics, and cognitive state of the AI.

Features:
- **Bot Status:** General health and operational status.
- **Cognitive Status:** Detailed insights into the Cognitive Brain's activity.
- **Recent Reflections:** Access to the latest learning insights, including LLM critiques.
- **Transaction Cost Analysis (TCA):** Metrics on execution quality and slippage.
- **Self-Model Description:** The AI's self-awareness of its strengths and weaknesses.

This dashboard provides a "window into the AI's mind," making it easier to
monitor, debug, and understand the complex behavior of the trading robot.

Usage:
    To run the FastAPI application:
    uvicorn web.main:app --reload --port 8000

    Then open your browser to http://localhost:8000/docs for the API documentation.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List
from datetime import datetime

# Import necessary components from the cognitive architecture
from cognitive.signal_processor import get_signal_processor
from cognitive.reflection_engine import get_reflection_engine
from cognitive.self_model import get_self_model
from execution.tca.transaction_cost_analyzer import get_tca_analyzer

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Kobe81 Traderbot Dashboard API",
    description="API for monitoring and introspecting the Kobe81 Traderbot's status and cognitive state.",
    version="1.0.0",
)

@app.get("/", response_class=HTMLResponse, summary="Home Page")
async def read_root():
    """
    Returns a simple HTML page with links to the API documentation and key endpoints.
    """
    html_content = """
    <html>
        <head>
            <title>Kobe81 Traderbot Dashboard</title>
            <link rel="icon" href="https://raw.githubusercontent.com/mljar/mljar-supervised/master/docs/img/mljar_logo.png" />
        </head>
        <body>
            <h1>Kobe81 Traderbot Dashboard</h1>
            <p>Welcome to the monitoring API for your intelligent trading robot.</p>
            <ul>
                <li><a href="/docs">API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc">Alternative API Documentation (ReDoc)</a></li>
                <li><a href="/status">General Bot Status</a></li>
                <li><a href="/cognitive_status">Cognitive System Status</a></li>
                <li><a href="/recent_reflections">Recent Reflections (Learning Log)</a></li>
                <li><a href="/recent_tca">Recent Transaction Cost Analysis (TCA)</a></li>
                <li><a href="/self_model_description">AI's Self-Description</a></li>
            </ul>
            <p>Use the API documentation to explore available endpoints.</p>
        </body>
    </html>
    """
    return html_content

@app.get("/status", summary="Get overall bot status")
async def get_bot_status() -> Dict[str, Any]:
    """
    Returns a high-level overview of the bot's operational status.
    """
    try:
        signal_processor = get_signal_processor()
        executor = signal_processor.brain # Assuming brain has executor status or expose via processor
        # This will be refined as more components are exposed
        status = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "operational",
            "kill_switch_active": False, # Placeholder
            "paper_mode": True, # Placeholder
            "active_cognitive_episodes": len(signal_processor._active_episodes),
            "log_level": logging.getLevelName(logger.getEffectiveLevel()),
        }
        return status
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cognitive_status", summary="Get cognitive system status")
async def get_cognitive_system_status() -> Dict[str, Any]:
    """
    Returns a detailed status of the CognitiveSignalProcessor and its underlying Brain.
    """
    try:
        processor = get_signal_processor()
        return processor.get_cognitive_status()
    except Exception as e:
        logger.error(f"Error getting cognitive status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recent_reflections", summary="Get recent learning reflections")
async def get_recent_reflections() -> List[Dict[str, Any]]:
    """
    Returns a list of the most recent learning reflections, including any LLM critiques.
    """
    try:
        reflection_engine = get_reflection_engine()
        # Returns a list of Reflection objects converted to dicts
        return [r.to_dict() for r in reflection_engine.get_recent_reflections(limit=10)]
    except Exception as e:
        logger.error(f"Error getting recent reflections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recent_tca", summary="Get recent Transaction Cost Analysis (TCA) metrics")
async def get_recent_tca() -> Dict[str, Any]:
    """
    Returns a summary of recent Transaction Cost Analysis (TCA) metrics.
    """
    try:
        tca_analyzer = get_tca_analyzer()
        return tca_analyzer.get_summary_tca_metrics(lookback_days=7)
    except Exception as e:
        logger.error(f"Error getting recent TCA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/self_model_description", summary="Get AI's self-description")
async def get_ai_self_description() -> Dict[str, str]:
    """
    Returns the AI's natural language self-description of its capabilities and limitations.
    """
    try:
        self_model = get_self_model()
        return {"self_description": self_model.get_self_description()}
    except Exception as e:
        logger.error(f"Error getting self-model description: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # This block is for local testing and development outside of uvicorn command.
    # In production, you would run with 'uvicorn web.main:app --host 0.0.0.0 --port 8000'
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
