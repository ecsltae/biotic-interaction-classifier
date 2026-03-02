#!/usr/bin/env python3
"""
MCP Server for Biotic Interaction Classification.

Exposes tools for validating and classifying biotic interaction sentences.
Uses stdio transport for integration with Claude Code.

Run directly: python classifier/tools/mcp_server.py
Or via .mcp.json configuration in Claude Code.
"""
import sys
import json
import logging
from pathlib import Path

# Add classifier root to path so we can import validator module
CLASSIFIER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CLASSIFIER_ROOT))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from validator.interaction_validator import (
    validate_interaction_sentence,
    batch_validate_sentences,
)

logger = logging.getLogger(__name__)
server = Server("metap-interaction-validator")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="validate_interaction",
            description="Validate whether a sentence describes a biotic interaction between organisms. Returns confidence score and reasoning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sentence": {
                        "type": "string",
                        "description": "The sentence to validate for biotic interaction content",
                    },
                    "use_llm": {
                        "type": "boolean",
                        "default": True,
                        "description": "Use LLM validation if ANTHROPIC_API_KEY is set, otherwise falls back to heuristics",
                    },
                },
                "required": ["sentence"],
            },
        ),
        Tool(
            name="batch_validate",
            description="Validate multiple sentences for biotic interactions. Returns only those above the confidence threshold.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sentences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of sentences to validate",
                    },
                    "min_confidence": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Minimum confidence threshold (0.0-1.0)",
                    },
                },
                "required": ["sentences"],
            },
        ),
        Tool(
            name="classify_text",
            description="Classify text using the trained ML ensemble API (BiomedBERT+RoBERTa). Requires the FastAPI to be running on port 8001.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to classify as biotic interaction or not",
                    }
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="get_training_stats",
            description="Get statistics about the current training dataset: row count, label distribution, latest version.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "validate_interaction":
        result = validate_interaction_sentence(
            arguments["sentence"],
            use_llm=arguments.get("use_llm", True),
        )
        return [TextContent(type="text", text=json.dumps(result.to_dict(), indent=2))]

    elif name == "batch_validate":
        results = batch_validate_sentences(
            arguments["sentences"],
            min_confidence=arguments.get("min_confidence", 0.5),
        )
        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    elif name == "classify_text":
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "http://localhost:8001/predict",
                    json={"text": arguments["text"]},
                    timeout=30,
                )
                return [TextContent(type="text", text=resp.text)]
        except httpx.ConnectError:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": "API not running",
                            "hint": "Start with: bash classifier/start_api.sh",
                        }
                    ),
                )
            ]
        except Exception as e:
            return [
                TextContent(type="text", text=json.dumps({"error": str(e)}))
            ]

    elif name == "get_training_stats":
        import pandas as pd

        data_dir = CLASSIFIER_ROOT / "data" / "training"
        csvs = sorted(data_dir.glob("training_data_globi_v*.csv"))
        if csvs:
            latest = csvs[-1]
            df = pd.read_csv(latest)
            stats = {
                "file": latest.name,
                "total_rows": len(df),
                "columns": list(df.columns),
            }
            if "label" in df.columns:
                stats["label_distribution"] = df["label"].value_counts().to_dict()
            elif "is_interaction" in df.columns:
                stats["label_distribution"] = (
                    df["is_interaction"].value_counts().to_dict()
                )
        else:
            stats = {"error": "No training data found in " + str(data_dir)}
        return [TextContent(type="text", text=json.dumps(stats, indent=2))]

    else:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}),
            )
        ]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
