from __future__ import annotations

import pytest

from immuno_mcp.core.cache import DiskCache
from immuno_mcp.core.errors import ErrorCode, ImmunoError
from immuno_mcp.core.fence import clean_text, clean_tree, sanitize_message
from immuno_mcp.core.logging import SecretRedactionFilter
from immuno_mcp.core.settings import Settings, profile_allows

ZWSP, ZWJ, RLO, TAG_A = chr(0x200B), chr(0x200D), chr(0x202E), chr(0xE0041)


def test_clean_text_strips_hidden_code_points_but_keeps_layout():
    hidden = f"a{ZWSP}b{RLO}c{TAG_A}d\x00e"
    assert clean_text(hidden) == "abcde"
    assert clean_text("line1\n\tline2\r\n") == "line1\n\tline2\r\n"


def test_clean_text_keeps_scientific_symbols():
    assert clean_text("ΔΔG 1.2 kcal/mol, 2.1 Å, α-helix") == "ΔΔG 1.2 kcal/mol, 2.1 Å, α-helix"


def test_sanitize_message_is_one_line_and_capped():
    out = sanitize_message(("bad\n\n   thing" + ZWSP + " happened ") * 50, max_chars=40)
    assert "\n" not in out and len(out) <= 40 and out.endswith("…")


def test_clean_tree_reaches_nested_values_and_keys():
    tree = {f"ti{ZWSP}tle": [f"a{RLO}b", {"x": f"c{ZWJ}d"}], "n": 3}
    assert clean_tree(tree) == {"title": ["ab", {"x": "cd"}], "n": 3}


def test_error_message_has_code_hint_and_retry_flag():
    err = ImmunoError(ErrorCode.RATE_LIMITED, "slow down", hint="wait a bit")
    assert err.retryable
    assert err.to_message() == "[RATE_LIMITED] slow down Hint: wait a bit Retryable."
    assert not ImmunoError(ErrorCode.NOT_FOUND, "nope").retryable


def test_settings_defaults_and_profile_order(tmp_path):
    s = Settings.from_env({"IMMUNOMCP_HOME": str(tmp_path)})
    assert s.profile == "lab" and s.max_spend_usd == 0.0 and s.japanfold_api_key is None
    assert profile_allows("compute", "lab") and not profile_allows("readonly", "lab")


def test_settings_rejects_unknown_profile_and_bad_numbers():
    with pytest.raises(ImmunoError):
        Settings.from_env({"IMMUNOMCP_PROFILE": "root"})
    with pytest.raises(ImmunoError):
        Settings.from_env({"IMMUNOMCP_MAX_SPEND_USD": "-1"})
    with pytest.raises(ImmunoError):
        Settings.from_env({"IMMUNOMCP_MAX_SPEND_USD": "lots"})


def test_secrets_are_hidden_in_repr_and_listed_for_redaction():
    s = Settings.from_env(
        {"JAPANFOLD_API_KEY": "jf_live_supersecretvalue", "HF_TOKEN": "hf_abcdefghijk"}
    )
    assert "supersecret" not in repr(s) and "hf_abcdefghijk" not in repr(s)
    assert s.key_present("JAPANFOLD_API_KEY") and not s.key_present("NCBI_API_KEY")
    assert set(s.secret_values()) == {"jf_live_supersecretvalue", "hf_abcdefghijk"}


def test_log_filter_redacts_configured_and_pattern_secrets():
    f = SecretRedactionFilter(["my-exact-secret-value"])
    text = "key my-exact-secret-value Authorization: Bearer abcdef1234567890 url?api_key=XYZ123 jf_live_abc123"
    out = f.redact(text)
    for leaked in ("my-exact-secret-value", "abcdef1234567890", "XYZ123", "jf_live_abc123"):
        assert leaked not in out


def test_cache_respects_ttl():
    c = DiskCache(":memory:")
    c.put("k", b"v", {"a": 1}, ttl=10, now=100.0)
    assert c.get("k", now=105.0) == (b"v", {"a": 1})
    assert c.get("k", now=111.0) is None
    assert c.stats() == {"entries": 0, "bytes": 0}
