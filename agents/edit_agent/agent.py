# ============================================================
# agents/edit_agent/agent.py
# Phase 5 — High-level async API for the Edit Agent
#
# Use this from the FastAPI backend or CLI.
# ============================================================

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator

from agents.edit_agent.graph import edit_graph
from agents.edit_agent.state import EditAgentState
from state_manager.manager import StateManager


class EditAgent:
    """
    High-level wrapper around the LangGraph edit graph.

    Usage
    -----
    agent = EditAgent()
    result = await agent.run_edit("Make the first scene darker")
    """

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())

    async def run_edit(
        self,
        query: str,
        current_version: str | None = None,
    ) -> dict:
        """
        Execute an edit command end-to-end.
        Returns the final state dict with result + new_version.
        """
        version = current_version or StateManager.latest_version() or "v1"

        initial_state: EditAgentState = {
            "raw_query":       query,
            "current_version": version,
            "intent":          "",
            "target":          "",
            "scope":           "",
            "parameters":      {},
            "plan":            [],
            "result":          {},
            "new_version":     "",
            "status":          "classifying",
            "errors":          [],
            "logs":            [],
        }

        config = {"configurable": {"thread_id": self.session_id}}
        final_state = await edit_graph.ainvoke(initial_state, config=config)
        return final_state

    async def stream_edit(
        self,
        query: str,
        current_version: str | None = None,
    ) -> AsyncIterator[dict]:
        """
        Stream intermediate node outputs.
        Yields dicts with {"node": str, "state": dict} for each step.
        """
        version = current_version or StateManager.latest_version() or "v1"

        initial_state: EditAgentState = {
            "raw_query":       query,
            "current_version": version,
            "intent":          "",
            "target":          "",
            "scope":           "",
            "parameters":      {},
            "plan":            [],
            "result":          {},
            "new_version":     "",
            "status":          "classifying",
            "errors":          [],
            "logs":            [],
        }

        config = {"configurable": {"thread_id": self.session_id}}

        async for event in edit_graph.astream(initial_state, config=config):
            for node_name, node_state in event.items():
                yield {"node": node_name, "state": node_state}
