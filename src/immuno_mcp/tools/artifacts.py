"""Bring files produced outside the server into the content-addressed workspace.

Filesystem paths are never accepted. ``import_artifact`` takes names RELATIVE to the inbox
directory (``<IMMUNOMCP_HOME>/inbox``) and opens them from an inbox directory descriptor with
``O_NOFOLLOW`` at every component, so a symbolic link anywhere in the name is refused and there
is no second path lookup after the open. Hard links (``st_nlink > 1``) and anything that is not a
regular file are refused too. The content is copied into the workspace through
``Workspace.store_bytes``; the inbox file is never modified, moved or deleted.

Import is all or nothing: every name and every content is checked before anything is stored.
"""

from __future__ import annotations

import asyncio
import codecs
import errno
import gzip
import hashlib
import io
import itertools
import os
import stat
import struct
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from .. import __version__
from ..analysis import structure as st
from ..core.envelope import Provenance, ResponseMode, ToolResult, envelope, to_result
from ..core.errors import ErrorCode, ImmunoError
from ..core.fence import clean_text, clean_tree
from .registry import ToolRegistrar
from .structure_local import MAX_ARTIFACT_BYTES

MAX_FILES = 25
MAX_NAME_CHARS = 300
MAX_SEGMENT_CHARS = 255
HEAD_BYTES = 256 * 1024
CHUNK = 1024 * 1024
NPZ_MAX_MEMBERS = 64
NPZ_MAX_CD_BYTES = 1_048_576  # central directory size; bounds what zipfile parses
MAX_GZIP_MEMBERS = 64
_EOCD_SEARCH = 65_557  # 22-byte end record plus the longest possible comment
LIST_MAX_FILES = 200
LIST_MAX_DEPTH = 4
LIST_MAX_SCAN = 10_000  # directory entries looked at in one listing, at most

_STRUCTURE_SUFFIXES = {".cif": "mmcif", ".mmcif": "mmcif", ".pdb": "pdb", ".ent": "pdb"}
_SUFFIX_KIND = {
    **_STRUCTURE_SUFFIXES,
    ".json": "json",
    ".npz": "npz",
    ".csv": "csv",
    ".tsv": "csv",
    ".fasta": "fasta",
    ".fa": "fasta",
    ".faa": "fasta",
}
_STORED_SUFFIX = {
    "mmcif": ".cif",
    "pdb": ".pdb",
    "json": ".json",
    "npz": ".npz",
    "csv": ".csv",
    "fasta": ".fasta",
}
_MIME = {
    "mmcif": "chemical/x-mmcif",
    "pdb": "chemical/x-pdb",
    "json": "application/json",
    "npz": "application/zip",
    "csv": "text/csv",
    "fasta": "text/plain",
}
SUPPORTED_HINT = (
    "Importable: mmCIF (.cif, .mmcif), PDB (.pdb, .ent), JSON (.json), numpy .npz, CSV/TSV "
    "(.csv, .tsv) and FASTA (.fasta, .fa, .faa); text kinds may be gzipped (.gz). The suffix "
    "must match the content."
)
UNVERIFIED = "source and job_id are what the caller declared; immuno-mcp did not verify them"
_BAD_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})


def _shown(name: str) -> str:
    """A given name as it may appear in a message: cleaned and short."""
    text = clean_text(name)
    return repr(text if len(text) <= 120 else text[:119] + "…")


def _has_bad_char(name: str) -> bool:
    return any(unicodedata.category(ch) in _BAD_CATEGORIES for ch in name)


def _bad_name(name: str, why: str) -> ImmunoError:
    return ImmunoError(
        ErrorCode.INVALID_INPUT,
        f"file name {_shown(name)} is not accepted: {why}",
        hint="Give names relative to the inbox, e.g. 'job123/model_0.cif'. Call import_artifact() "
        "without files to list the inbox.",
    )


def validate_names(names: list[str]) -> list[tuple[str, list[str]]]:
    """(name, segments) for each distinct name, in order. Raises before any I/O."""
    out: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name:
            raise _bad_name(str(name), "it is empty")
        if len(name) > MAX_NAME_CHARS:
            raise _bad_name(name, f"it is longer than {MAX_NAME_CHARS} characters")
        if "\x00" in name:
            raise _bad_name(name, "it contains a NUL character")
        if "\\" in name:
            raise _bad_name(name, "it contains a backslash; use '/' between folders")
        if name.startswith("/"):
            raise _bad_name(name, "absolute paths are not accepted, only names inside the inbox")
        if _has_bad_char(name):
            raise _bad_name(name, "it contains control, format or surrogate characters")
        segments = name.split("/")
        for seg in segments:
            if seg in ("", ".", ".."):
                raise _bad_name(name, "empty, '.' and '..' path segments are not accepted")
            if len(seg) > MAX_SEGMENT_CHARS:
                raise _bad_name(name, f"a segment is longer than {MAX_SEGMENT_CHARS} characters")
        if name in seen:
            continue
        seen.add(name)
        out.append((name, segments))
    return out


# ---- secure file access --------------------------------------------------------------------


def _require_posix() -> None:
    if (
        os.open not in os.supports_dir_fd
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise ImmunoError(
            ErrorCode.NOT_CONFIGURED,
            "import_artifact needs a POSIX system (openat with O_NOFOLLOW); it is not "
            "available on this platform",
            hint="Run immuno-mcp on Linux or macOS to import files.",
        )


def _dir_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _file_flags() -> int:
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def _os_error(exc: OSError, shown: str) -> ImmunoError:
    """A typed error from errno and strerror only; str(exc) holds paths and is never used."""
    reason = exc.strerror or errno.errorcode.get(exc.errno or 0, "error")
    code = exc.errno
    listing = "Call import_artifact() without files to list the inbox."
    if code in (errno.ENOENT, errno.ENOTDIR):
        return ImmunoError(ErrorCode.NOT_FOUND, f"{shown} is not in the inbox", hint=listing)
    if code == errno.ELOOP or code == getattr(errno, "EMLINK", None):
        return ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"{shown}: symbolic links are not followed",
            hint="Copy the file itself into the inbox instead of linking to it.",
        )
    if code in (errno.EACCES, errno.EPERM):
        return ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"{shown} cannot be read ({reason})",
            hint="Make the file readable by the user that runs immuno-mcp.",
        )
    if code == errno.ENXIO:
        return ImmunoError(
            ErrorCode.INVALID_INPUT, f"{shown} is not a regular file", hint=SUPPORTED_HINT
        )
    return ImmunoError(ErrorCode.INTERNAL, f"{shown} could not be read ({reason})")


def _open_root(inbox: Path) -> int:
    try:
        inbox.mkdir(parents=True, exist_ok=True, mode=0o700)
        return os.open(inbox, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as exc:
        reason = exc.strerror or "error"
        raise ImmunoError(
            ErrorCode.NOT_CONFIGURED,
            f"the inbox directory cannot be opened ({reason})",
            hint="Check that <IMMUNOMCP_HOME>/inbox is a readable directory.",
        ) from None


def _is_link(name: str, dir_fd: int) -> bool:
    try:
        return stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
    except OSError:
        return False


def _open_in_inbox(root_fd: int, segments: list[str]) -> int:
    """Descriptor of a file below root_fd. No component may be a symbolic link."""
    cur = root_fd
    owned = False
    try:
        for seg in segments[:-1]:
            try:
                nxt = os.open(seg, _dir_flags(), dir_fd=cur)
            except OSError as exc:
                # Linux reports a symlinked folder as ENOTDIR here; say what it really is.
                # lstat does not follow the link and its result only picks the message.
                if exc.errno == errno.ENOTDIR and _is_link(seg, cur):
                    raise OSError(errno.ELOOP, os.strerror(errno.ELOOP)) from None
                raise
            if owned:
                os.close(cur)
            cur, owned = nxt, True
        return os.open(segments[-1], _file_flags(), dir_fd=cur)
    finally:
        if owned:
            os.close(cur)


def _read_fd(fd: int, shown: str, budget: int) -> bytes:
    """Read a regular, singly linked file from its descriptor, at most `budget` bytes."""
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"{shown} is not a regular file",
            hint="Only regular files are imported (no folders, FIFOs, sockets or devices).",
        )
    if info.st_nlink != 1:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"{shown} has {info.st_nlink} hard links; hard-linked files are refused",
            hint="Copy the file into the inbox (cp) instead of linking it (ln).",
        )
    if info.st_size > MAX_ARTIFACT_BYTES:
        raise ImmunoError(ErrorCode.TOO_LARGE, f"{shown} is larger than 200 MB")
    if info.st_size > budget:
        raise ImmunoError(
            ErrorCode.TOO_LARGE,
            f"{shown}: the files of one call add up to more than 200 MB",
            hint="Import the files in several calls.",
        )
    chunks: list[bytes] = []
    total = 0
    while True:
        block = os.read(fd, min(CHUNK, budget + 1 - total))
        if not block:
            break
        chunks.append(block)
        total += len(block)
        if total > budget:
            raise ImmunoError(
                ErrorCode.TOO_LARGE,
                f"{shown} grew while it was read and is now over the limit",
                hint="Wait until the download has finished, then import again.",
            )
    return b"".join(chunks)


# ---- classification ------------------------------------------------------------------------


@dataclass
class Prepared:
    name: str
    data: bytes  # as stored (still gzipped when the file was gzipped)
    kind: str
    gzip: bool


def _gunzip_head(data: bytes, shown: str) -> bytes:
    """First HEAD_BYTES of the unpacked data; the whole stream is checked against the cap.

    GzipFile reads multi-member files and treats trailing bytes like gzip.decompress does
    (zero padding is skipped, anything else is an error), so the bound holds for the same input
    that read_artifact later unpacks.
    """
    head = bytearray()
    total = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
            while True:
                block = gz.read(CHUNK)
                if not block:
                    break
                total += len(block)
                if total > MAX_ARTIFACT_BYTES:
                    raise ImmunoError(
                        ErrorCode.TOO_LARGE,
                        f"{shown} unpacks to more than 200 MB",
                        hint="Import a smaller file.",
                    )
                if len(head) < HEAD_BYTES:
                    head += block[: HEAD_BYTES - len(head)]
    except (gzip.BadGzipFile, EOFError, zlib.error):
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"{shown} is a broken or truncated gzip file",
            hint="Check that the download finished, or import the unpacked file.",
        ) from None
    return bytes(head)


def _suffix_of(name: str) -> str:
    last = name.rsplit("/", 1)[-1].lower()
    if last.endswith(".gz"):
        last = last[:-3]
    return os.path.splitext(last)[1]


def _mismatch(shown: str, suffix: str, what: str) -> ImmunoError:
    return ImmunoError(
        ErrorCode.INVALID_INPUT,
        f"{shown} has the suffix {suffix} but its content is not {what}",
        hint=SUPPORTED_HINT,
    )


def _check_eocd(data: bytes, shown: str) -> None:
    """Bound the zip central directory BEFORE zipfile parses all of it.

    Uses the same end-of-central-directory record that zipfile picks (the last signature in
    the final 65,557 bytes). The entry count there can lie; the central directory byte size
    is what bounds the parse, and the member checks after parsing still apply.
    """
    hint = "Import the numpy .npz written by np.savez; zip64 and very large archives are refused."
    pos = data.rfind(b"PK\x05\x06", max(0, len(data) - _EOCD_SEARCH))
    if pos < 0 or len(data) - pos < 22:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT, f"{shown} is a broken zip archive (no end record)", hint=hint
        )
    _sig, _disk, _cd_disk, _n_disk, n_total, cd_size, cd_offset, _clen = struct.unpack_from(
        "<4s4H2LH", data, pos
    )
    if (
        (pos >= 20 and data[pos - 20 : pos - 16] == b"PK\x06\x07")
        or n_total == 0xFFFF
        or cd_size == 0xFFFFFFFF
        or cd_offset == 0xFFFFFFFF
    ):
        raise ImmunoError(
            ErrorCode.INVALID_INPUT, f"{shown} is a zip64 archive, which is not imported", hint=hint
        )
    if n_total > NPZ_MAX_MEMBERS:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"{shown} has {n_total} members; at most {NPZ_MAX_MEMBERS} are accepted",
            hint=hint,
        )
    if cd_size > NPZ_MAX_CD_BYTES:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"{shown} has a zip central directory larger than 1 MiB",
            hint=hint,
        )


def _utf8_ok(head: bytes, complete: bool) -> bool:
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head, final=complete)
    except UnicodeDecodeError:
        return False
    return True


def classify(name: str, data: bytes) -> Prepared:
    """Decide the kind from the suffix AND the content. Raises INVALID_INPUT on a mismatch."""
    shown = _shown(name)
    suffix = _suffix_of(name)
    kind = _SUFFIX_KIND.get(suffix)
    if kind is None:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"{shown} is not an importable kind of file",
            hint=SUPPORTED_HINT,
        )
    is_gzip = data[:2] == b"\x1f\x8b"
    if kind == "npz":
        if is_gzip:
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"{shown} is a gzipped .npz; numpy archives are imported unpacked",
                hint="Unpack it (gunzip) and import the .npz itself.",
            )
        if not data.startswith(b"PK\x03\x04"):
            raise _mismatch(shown, suffix, "a numpy .npz (zip) archive")
        _check_eocd(data, shown)
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:  # reads the central directory only
                members = zf.infolist()
        except (zipfile.BadZipFile, zipfile.LargeZipFile, ValueError, EOFError):
            raise ImmunoError(
                ErrorCode.INVALID_INPUT, f"{shown} is a broken zip archive", hint=SUPPORTED_HINT
            ) from None
        if len(members) > NPZ_MAX_MEMBERS:
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"{shown} has {len(members)} members; at most {NPZ_MAX_MEMBERS} are accepted",
            )
        if sum(m.file_size for m in members) > MAX_ARTIFACT_BYTES:
            raise ImmunoError(ErrorCode.TOO_LARGE, f"{shown} unpacks to more than 200 MB")
        return Prepared(name=name, data=data, kind="npz", gzip=False)

    if is_gzip:
        # Every member of a valid stream starts with these bytes and the pattern cannot overlap
        # itself, so the count never undercounts members. gzip.decompress (used later by
        # read_artifact) is quadratic in the member count, hence the limit.
        if data.count(b"\x1f\x8b\x08") > MAX_GZIP_MEMBERS:
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"{shown} is a gzip file made of more than {MAX_GZIP_MEMBERS} members (for "
                "example bgzip output); such files are not supported",
                hint="Recompress it as one member, e.g. gunzip -c file.gz | gzip -c > new.gz "
                "(or gzip -c file > file.gz), then import again.",
            )
        head = _gunzip_head(data, shown)
        complete = len(head) < HEAD_BYTES
    else:
        head = data[:HEAD_BYTES]
        complete = len(data) <= HEAD_BYTES

    if kind in ("mmcif", "pdb"):
        sniffed = st.sniff_format(head.decode("utf-8", errors="replace"))
        wanted = "cif" if kind == "mmcif" else "pdb"
        if sniffed != wanted:
            raise _mismatch(shown, suffix, "mmCIF" if kind == "mmcif" else "a PDB coordinate file")
    elif kind == "json":
        if head.lstrip()[:1] not in (b"{", b"["):
            raise _mismatch(shown, suffix, "JSON (an object or an array)")
    elif kind == "csv":
        if b"\x00" in head or not _utf8_ok(head, complete):
            raise _mismatch(shown, suffix, "UTF-8 delimited text")
    elif kind == "fasta":
        text = head.decode("utf-8", errors="replace")
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if not first.startswith(">"):
            raise _mismatch(shown, suffix, "FASTA (the first line must start with '>')")
    return Prepared(name=name, data=data, kind=kind, gzip=is_gzip)


def prepare_all(inbox: Path, items: list[tuple[str, list[str]]]) -> list[Prepared]:
    """Phase 1: open, read and check every file. Nothing is written."""
    root = _open_root(inbox)
    prepared: list[Prepared] = []
    budget = MAX_ARTIFACT_BYTES
    try:
        for name, segments in items:
            shown = _shown(name)
            try:
                fd = _open_in_inbox(root, segments)
            except OSError as exc:
                raise _os_error(exc, shown) from None
            try:
                data = _read_fd(fd, shown, budget)
            except OSError as exc:
                raise _os_error(exc, shown) from None
            finally:
                os.close(fd)
            budget -= len(data)
            prepared.append(classify(name, data))
    finally:
        os.close(root)
    return prepared


# ---- listing -------------------------------------------------------------------------------


def _importable(name: str) -> bool:
    low = name.lower()
    if low.endswith(".npz.gz"):
        return False
    return _suffix_of(low) in _SUFFIX_KIND


def _listable(name: str) -> bool:
    if name.startswith(".") or name.endswith(".part") or _has_bad_char(name):
        return False
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def list_inbox(inbox: Path) -> dict[str, Any]:
    root = _open_root(inbox)
    files: list[dict[str, Any]] = []
    counts = {"total": 0, "skipped": 0, "scanned": 0}
    truncated = False

    def walk(fd: int, prefix: str, depth: int) -> None:
        nonlocal truncated
        with os.scandir(fd) as it:
            room = LIST_MAX_SCAN - counts["scanned"]
            names = sorted(e.name for e in itertools.islice(it, room + 1))
        if len(names) > room:
            truncated = True
            names = names[:room]
        counts["scanned"] += len(names)
        for name in names:
            if not _listable(name):
                counts["skipped"] += 1
                continue
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except OSError:
                counts["skipped"] += 1
                continue
            if stat.S_ISDIR(info.st_mode):
                if depth >= LIST_MAX_DEPTH:
                    counts["skipped"] += 1
                    continue
                try:
                    sub = os.open(name, _dir_flags(), dir_fd=fd)
                except OSError:
                    counts["skipped"] += 1
                    continue
                try:
                    walk(sub, f"{prefix}{name}/", depth + 1)
                finally:
                    os.close(sub)
            elif stat.S_ISREG(info.st_mode):
                counts["total"] += 1
                if len(files) < LIST_MAX_FILES:
                    rel = f"{prefix}{name}"
                    files.append(
                        {
                            "name": rel,
                            "bytes": info.st_size,
                            "importable": _importable(rel) and info.st_nlink == 1,
                        }
                    )
                else:
                    truncated = True
            else:  # symbolic links, FIFOs, sockets, devices
                counts["skipped"] += 1

    try:
        walk(root, "", 0)
    finally:
        os.close(root)
    files.sort(key=lambda f: f["name"])
    return {
        "inbox": str(inbox),
        "files": files,
        "total": counts["total"],
        "skipped": counts["skipped"],
        "truncated": truncated,
    }


# ---- tool ----------------------------------------------------------------------------------


def _next_steps(rows: list[dict[str, Any]]) -> list[str]:
    kinds = {r["kind"] for r in rows}
    structures = [r["artifact_id"] for r in rows if r["kind"] in ("mmcif", "pdb")]
    confidences = [r["artifact_id"] for r in rows if r["kind"] in ("json", "npz")]
    steps: list[str] = []
    if structures:
        steps.append(
            f"extract_chains(artifact_id='{structures[0]}') to see the chains and sequences."
        )
        steps.append(
            "interface_contacts or map_epitope_paratope(artifact_id=..., ...) for the interface."
        )
    if confidences:
        model = f"'{structures[0]}'" if structures else "<the model's artifact id>"
        steps.append(
            f"compute_ipsae(pae_artifact_id='{confidences[0]}', group_a=[...], group_b=[...], "
            f"structure_artifact_id={model}). It reads AlphaFold 3, ColabFold, AlphaFold DB, "
            "Boltz-2 and JapanFold confidence files; a confidence file in another format "
            "imports fine but compute_ipsae may refuse it."
        )
    if kinds & {"csv", "fasta"}:
        steps.append("CSV and FASTA artifacts are stored; read their contents with your own tools.")
    return steps


def register(reg: ToolRegistrar) -> None:
    app = reg.app

    @reg.tool(
        group="workspace",
        profile="lab",
        read_only=False,
        idempotent=True,
        destructive=False,
        open_world=False,
    )
    async def import_artifact(
        files: Annotated[
            list[str] | None,
            Field(
                max_length=MAX_FILES,
                description="File names RELATIVE to the inbox, e.g. ['job123/model_0.cif', "
                "'job123/confidences.json']. Never a filesystem path. Omit to list the inbox.",
            ),
        ] = None,
        source: Annotated[
            str | None,
            Field(
                max_length=200,
                description="Where the files came from, e.g. 'OpenFold3 via Foundry CLI'. "
                "Stored as declared, not verified.",
            ),
        ] = None,
        job_id: Annotated[
            str | None,
            Field(max_length=100, description="The external job id, stored as declared"),
        ] = None,
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """Copy files made outside immuno-mcp from the inbox into the workspace, by name.

        Use it when a structure or confidence file was made elsewhere and you want to analyse it
        with extract_chains, interface_contacts, map_epitope_paratope or compute_ipsae, which
        take workspace artifact ids. Files are put into the inbox (<IMMUNOMCP_HOME>/inbox) with
        the shell, for example a CLI download with --output <inbox>/<job id>.

        Without `files` it LISTS the inbox (names, sizes, whether each is importable). With
        `files` it imports those names, which are relative to the inbox. Filesystem paths are
        never accepted: absolute paths, '..', backslashes and control characters are refused,
        and symbolic links (anywhere in the name) and hard-linked files are refused. Only
        regular files of these kinds are imported, checked by suffix AND content: mmCIF, PDB,
        JSON, numpy .npz, CSV/TSV and FASTA; text kinds may be gzipped. At most 25 files and
        200 MB per call. If any file is refused, nothing is imported.

        The artifact id is the sha256 of the stored bytes; importing the same content again
        returns the same id. `source` and `job_id` are recorded as declared and are not
        verified. compute_ipsae reads AlphaFold 3, ColabFold, AlphaFold DB, Boltz-2 and
        JapanFold confidence files; a confidence file in another format imports fine but
        compute_ipsae may refuse it. The inbox files are left untouched.
        Next: extract_chains(artifact_id=...) on a model, then interface_contacts or
        map_epitope_paratope; compute_ipsae(pae_artifact_id=..., structure_artifact_id=...).
        """
        workspace = app.workspace
        if workspace.artifacts_dir is None:
            raise ImmunoError(
                ErrorCode.NOT_CONFIGURED,
                "this workspace has no artifacts directory, so files cannot be imported",
                hint="Run the server with a workspace on disk (IMMUNOMCP_HOME).",
            )
        _require_posix()
        inbox = app.settings.inbox_dir

        if not files:
            listing = await asyncio.to_thread(list_inbox, inbox)
            warnings = []
            if listing["skipped"]:
                warnings.append(
                    f"{listing['skipped']} entries were skipped (symbolic links, hidden or "
                    "partial files, special files, unreadable names or folders too deep)."
                )
            return to_result(
                envelope(
                    clean_tree(listing),
                    provenance=Provenance(
                        source="workspace_inbox",
                        version=f"immuno-mcp {__version__}",
                        query={"listing": True},
                    ),
                    warnings=warnings,
                    untrusted=["files[].name"],
                    next_steps=[
                        "import_artifact(files=['<name>', ...], source=..., job_id=...) to copy "
                        "importable files into the workspace.",
                        "Files are put into the inbox with the shell, for example a CLI download "
                        "with --output <inbox>/<job id>.",
                    ],
                ),
                response_mode,
            )

        items = validate_names(files)
        declared_source = clean_text(source) if source else None
        declared_job = clean_text(job_id) if job_id else None
        label = f"imported: {declared_source}" if declared_source else "imported"

        prepared = await asyncio.to_thread(prepare_all, inbox, items)

        def store_all() -> tuple[list[dict[str, Any]], list[Provenance], list[str]]:
            rows: list[dict[str, Any]] = []
            provs: list[Provenance] = []
            warns: list[str] = []
            for p in prepared:
                digest = hashlib.sha256(p.data).hexdigest()
                existing = workspace.get_artifact(digest)
                suffix = _STORED_SUFFIX[p.kind] + (".gz" if p.gzip else "")
                try:
                    row = workspace.store_bytes(
                        p.data,
                        kind=p.kind,
                        suffix=suffix,
                        mime="application/gzip" if p.gzip else _MIME.get(p.kind),
                        source=label,
                        job_id=declared_job,
                    )
                except OSError as exc:
                    reason = exc.strerror or "error"
                    raise ImmunoError(
                        ErrorCode.INTERNAL,
                        f"{_shown(p.name)} could not be written to the workspace ({reason})",
                        hint="Check free disk space and permissions of the workspace folder. "
                        "Files listed before this one may already be stored.",
                    ) from None
                if existing is not None and (
                    existing.get("source") != label or existing.get("job_id") != declared_job
                ):
                    warns.append(
                        f"{_shown(p.name)} was already in the workspace; the source and job_id "
                        "of the first import stand."
                    )
                rows.append(
                    {
                        "name": p.name,
                        "artifact_id": row["id"],
                        "kind": p.kind,
                        "bytes": len(p.data),
                        "gzip": p.gzip,
                        "already_in_workspace": existing is not None,
                    }
                )
                provs.append(
                    Provenance(
                        source="workspace_import",
                        sha256=digest,
                        version=f"immuno-mcp {__version__}",
                        query=clean_tree(
                            {
                                "file": p.name,
                                "declared_source": declared_source,
                                "declared_job_id": declared_job,
                            }
                        ),
                    )
                )
            return rows, provs, warns

        rows, provs, warns = await asyncio.to_thread(store_all)
        data = clean_tree(
            {
                "inbox": str(inbox),
                "imported": rows,
                "declared": {"source": declared_source, "job_id": declared_job},
            }
        )
        return to_result(
            envelope(
                data,
                provenance=provs,
                warnings=[UNVERIFIED, *warns],
                untrusted=["imported[].name", "files[].name", "declared.source", "declared.job_id"],
                next_steps=_next_steps(rows),
            ),
            response_mode,
        )
