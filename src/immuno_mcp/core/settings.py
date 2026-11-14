"""Runtime settings, read once from the environment. Secrets never appear in repr()."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .errors import ErrorCode, ImmunoError

Profile = Literal["readonly", "lab", "compute"]
PROFILE_RANK: dict[str, int] = {"readonly": 0, "lab": 1, "compute": 2}
PROFILE_HELP = {
    "readonly": "public reads and pure local analysis; no state is written",
    "lab": "adds workspace writes (candidates, scores, downloaded files); cannot spend money",
    "compute": "adds paid compute (predictions, designs); needs a key and a spend cap above zero",
}


def profile_allows(active: str, required: str) -> bool:
    return PROFILE_RANK[active] >= PROFILE_RANK[required]


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ImmunoError(ErrorCode.INVALID_INPUT, f"{name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise ImmunoError(ErrorCode.INVALID_INPUT, f"{name} must not be negative")
    return value


@dataclass(frozen=True)
class Settings:
    home: Path
    profile: Profile = "lab"
    max_spend_usd: float = 0.0
    contact: str | None = None
    japanfold_base_url: str = "https://api.japanfold.aiand.com"
    japanfold_api_key: str | None = field(default=None, repr=False)
    hf_token: str | None = field(default=None, repr=False)
    ncbi_api_key: str | None = field(default=None, repr=False)
    offline: bool = False
    http_timeout: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        profile = e.get("IMMUNOMCP_PROFILE", "lab").strip().lower() or "lab"
        if profile not in PROFILE_RANK:
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"IMMUNOMCP_PROFILE must be one of {sorted(PROFILE_RANK)}, got {profile!r}",
            )
        home = Path(e.get("IMMUNOMCP_HOME") or Path.home() / ".immuno-mcp").expanduser()
        return cls(
            home=home,
            profile=profile,  # type: ignore[arg-type]
            max_spend_usd=_float(e, "IMMUNOMCP_MAX_SPEND_USD", 0.0),
            contact=e.get("IMMUNOMCP_CONTACT") or None,
            japanfold_base_url=(e.get("JAPANFOLD_BASE_URL") or cls.japanfold_base_url).rstrip("/"),
            japanfold_api_key=e.get("JAPANFOLD_API_KEY") or None,
            hf_token=e.get("HF_TOKEN") or None,
            ncbi_api_key=e.get("NCBI_API_KEY") or None,
            offline=e.get("IMMUNOMCP_OFFLINE", "").lower() in {"1", "true", "yes"},
            http_timeout=_float(e, "IMMUNOMCP_HTTP_TIMEOUT", 30.0) or 30.0,
        )

    @property
    def cache_path(self) -> Path:
        return self.home / "cache" / "http.sqlite"

    @property
    def workspace_path(self) -> Path:
        return self.home / "workspace" / "workspace.sqlite"

    @property
    def artifacts_dir(self) -> Path:
        return self.home / "workspace" / "artifacts"

    @property
    def inbox_dir(self) -> Path:
        """Where files made outside the server are put for import_artifact (names only)."""
        return self.home / "inbox"

    def secret_values(self) -> list[str]:
        """Every configured secret, so log filters can redact them."""
        return [s for s in (self.japanfold_api_key, self.hf_token, self.ncbi_api_key) if s]

    def key_present(self, env_name: str) -> bool:
        return {
            "JAPANFOLD_API_KEY": self.japanfold_api_key,
            "HF_TOKEN": self.hf_token,
            "NCBI_API_KEY": self.ncbi_api_key,
        }.get(env_name) is not None
