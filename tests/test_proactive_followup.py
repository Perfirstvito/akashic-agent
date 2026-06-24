from __future__ import annotations

from agent.core.proactive_followup import resolve_proactive_followup


def test_resolve_proactive_followup_second_item_from_recent_source_refs():
    messages = [
        {
            "id": "cli:1:0",
            "role": "assistant",
            "content": "推送了两篇论文",
            "proactive": True,
            "source_refs": [
                {
                    "id": "feed:p1",
                    "kind": "content",
                    "source_name": "arXiv",
                    "title": "First Paper",
                    "url": "https://arxiv.org/abs/2601.00001",
                },
                {
                    "id": "feed:p2",
                    "kind": "content",
                    "source_name": "arXiv",
                    "title": "Second Paper",
                    "url": "https://arxiv.org/abs/2601.00002",
                },
            ],
        }
    ]

    result = resolve_proactive_followup("下面那篇给我讲讲", messages)

    assert result is not None
    assert result["proactive_followup"] is True
    assert result["proactive_followup_status"] == "resolved"
    assert result["resolved_from_proactive_message_id"] == "cli:1:0"
    assert result["resolved_proactive_refs"] == [
        {
            "id": "feed:p2",
            "kind": "content",
            "source_name": "arXiv",
            "title": "Second Paper",
            "url": "https://arxiv.org/abs/2601.00002",
        }
    ]
    assert "Second Paper" in result["resolved_proactive_hint"]


def test_resolve_proactive_followup_marks_ambiguous_batch_when_no_ordinal():
    messages = [
        {
            "id": "cli:1:0",
            "role": "assistant",
            "content": "推送了两篇论文",
            "proactive": True,
            "source_refs": [
                {"id": "feed:p1", "title": "First Paper"},
                {"id": "feed:p2", "title": "Second Paper"},
            ],
        }
    ]

    result = resolve_proactive_followup("刚才那篇给我讲讲", messages)

    assert result is not None
    assert result["proactive_followup_status"] == "ambiguous"
    assert [ref["id"] for ref in result["resolved_proactive_refs"]] == ["feed:p1", "feed:p2"]


def test_resolve_proactive_followup_ignores_unrelated_message():
    messages = [
        {
            "role": "assistant",
            "content": "推送了一篇论文",
            "proactive": True,
            "source_refs": [{"id": "feed:p1", "title": "First Paper"}],
        }
    ]

    assert resolve_proactive_followup("今天吃什么", messages) is None
