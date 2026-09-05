"""
Real-Time Event Management & Server-Sent Events (SSE) Bus for NeuroTract 2.0

Enables live, event-driven streaming of analysis milestones, stage transitions,
and genuine scientific telemetry directly to web clients.
"""

from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class JobEventManager:
    """
    In-memory pub-sub event broadcaster with history buffering and keep-alive heartbeats.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}

    def emit(self, job_id: str, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Broadcast an event to all subscribers and append to historical replay log"""
        payload = {
            "job_id": job_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if data:
            payload.update(data)

        if job_id not in self._history:
            self._history[job_id] = []
        self._history[job_id].append(payload)

        # Dispatch to active subscribers
        queues = self._subscribers.get(job_id, [])
        for q in queues:
            try:
                loop = getattr(q, '_loop', None)
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(q.put_nowait, payload)
                else:
                    q.put_nowait(payload)
            except Exception as e:
                logger.debug(f"Failed to put event on queue for job {job_id}: {e}")

    def get_history(self, job_id: str) -> List[Dict[str, Any]]:
        """Get list of past events for a job"""
        return self._history.get(job_id, [])

    async def event_generator(self, job_id: str) -> AsyncGenerator[str, None]:
        """
        Yield Server-Sent Events (SSE) formatted text stream with historical replay
        and periodic keep-alive comments.
        """
        queue: asyncio.Queue = asyncio.Queue()
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        self._subscribers[job_id].append(queue)

        try:
            # 1. Replay historical events
            past_events = self._history.get(job_id, [])
            for event in past_events:
                yield f"data: {json.dumps(event)}\n\n"

            # If the job has already finished, close the stream
            if past_events and past_events[-1].get("event_type") in ("job_completed", "job_failed", "job_cancelled"):
                return

            # 2. Stream live events
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("event_type") in ("job_completed", "job_failed", "job_cancelled"):
                        break
                except asyncio.TimeoutError:
                    # Keep connection alive through proxies/NAT
                    yield f": keepalive {datetime.utcnow().isoformat()}\n\n"
        finally:
            if job_id in self._subscribers and queue in self._subscribers[job_id]:
                self._subscribers[job_id].remove(queue)


# Global singleton instance
global_event_manager = JobEventManager()
