from __future__ import annotations

from proactive_v2.rules import collect_content_rule_keys, read_dynamic_rule_context


def test_collect_content_rule_keys_uses_type_subscription_and_explicit_keys():
    keys = collect_content_rule_keys([
        {
            "source_type": "arXiv",
            "subscription_type": "arxiv_cs.AI",
            "subscription_id": "arxiv:cs.AI",
            "_rule_keys": ["content/papers", "subscriptions/custom"],
        }
    ])

    assert keys == [
        "content",
        "content/arxiv",
        "content/arxiv-cs-ai",
        "subscriptions/arxiv-cs-ai",
        "content/papers",
        "subscriptions/custom",
    ]


def test_collect_content_rule_keys_falls_back_to_source_and_url():
    keys = collect_content_rule_keys([
        {
            "source": "B站 某UP",
            "url": "https://www.bilibili.com/video/BV1",
        },
        {
            "source": "Paper",
            "url": "https://arxiv.org/abs/2601.00001",
        },
    ])

    assert keys == ["content", "content/bilibili", "content/arxiv"]


def test_read_dynamic_rule_context_reads_existing_rule_files_only(tmp_path):
    (tmp_path / "proactive_rules" / "content").mkdir(parents=True)
    (tmp_path / "proactive_rules" / "subscriptions").mkdir(parents=True)
    (tmp_path / "proactive_rules" / "content.md").write_text("global content rule", encoding="utf-8")
    (tmp_path / "proactive_rules" / "content" / "arxiv.md").write_text("arxiv rule", encoding="utf-8")
    (tmp_path / "proactive_rules" / "subscriptions" / "arxiv-cs-ai.md").write_text(
        "subscription rule",
        encoding="utf-8",
    )

    context = read_dynamic_rule_context(
        tmp_path,
        [
            {
                "source_type": "arxiv",
                "subscription_id": "arxiv-cs-ai",
            }
        ],
    )

    assert "[proactive_rules/content.md]" in context
    assert "global content rule" in context
    assert "[proactive_rules/content/arxiv.md]" in context
    assert "arxiv rule" in context
    assert "[proactive_rules/subscriptions/arxiv-cs-ai.md]" in context
    assert "subscription rule" in context


def test_read_dynamic_rule_context_sanitizes_rule_keys(tmp_path):
    (tmp_path / "proactive_rules" / "content").mkdir(parents=True)
    (tmp_path / "proactive_rules" / "content" / "safe.md").write_text("safe rule", encoding="utf-8")

    context = read_dynamic_rule_context(
        tmp_path,
        [{"_rule_keys": ["../secrets", "content/safe"]}],
    )

    assert "safe rule" in context
    assert "secrets" not in context
