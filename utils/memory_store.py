"""Markdown-based long-term memory store.

Replaces the prior JSONL+FAISS implementation with an in-context retrieval
design inspired by Claude Code (CLAUDE.md), Cline Memory Bank, and
MemGPT/Letta core memory.

Storage layout (under <store_path>/, which is a directory, not a file):

    MEMORY.md              — index, lists all memory files
    domain_facts.md        — verified facts (papers, companies, people)
    search_strategies.md   — query/retrieval patterns that work
    source_credibility.md  — per-domain trust ratings
    failed_patterns.md     — anti-patterns to avoid
    user_preferences.md    — observed user preferences

Reading: every file is concatenated and returned as a single MemoryBundle
whose `format_context()` is injected into the Planner system prompt at task
start. There is no embedding-based retrieval — the consuming LLM picks
relevant pieces in-context.

Writing: at task end, a memory_writer LLM examines current memory + the new
task's findings and outputs APPEND/REPLACE/NOOP actions. Semantic
deduplication is the LLM's job, not an embedding similarity threshold.

Backward compatibility:
- The legacy `MemoryEntry` class is kept as a stub so old imports do not
  break, but it has no functional role anymore.
- If `store_path` ends with `.jsonl` (legacy config), the parent directory
  is used as the new MD store directory.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Markdown templates
# ═══════════════════════════════════════════════════════════════════════

_INDEX_TEMPLATE = """\
# Long-Term Memory Index

This file is the entry point of the agent's long-term memory.
Each line below is a pointer to a memory file in this directory.

- [Domain Facts](domain_facts.md) — verified facts (papers, companies, people, dates)
- [Search Strategies](search_strategies.md) — query/retrieval patterns that work well
- [Source Credibility](source_credibility.md) — trust ratings of URLs and domains
- [Failed Patterns](failed_patterns.md) — anti-patterns to avoid (404s, low-quality sources)
- [User Preferences](user_preferences.md) — observed user preferences across tasks
"""

_FILE_TEMPLATES: Dict[str, str] = {
    "domain_facts.md": """\
---
type: domain_facts
last_updated: never
---

# Domain Facts

Verified facts the agent has discovered. Each entry should cite its source
where possible.

## Papers

## Companies

## People
""",
    "search_strategies.md": """\
---
type: search_strategies
last_updated: never
---

# Search Strategies

Validated query/retrieval patterns. Each entry describes a class of
information and the search approach that works best for it.

## arXiv lookups

## Company information

## Author profiles

## Failure patterns to avoid
""",
    "source_credibility.md": """\
---
type: source_credibility
last_updated: never
---

# Source Credibility

Per-domain trust ratings derived from multi-source verification history.

## High trust (multi-source confirmed)

## Medium trust

## Low trust (frequent errors)

## Dead / abandoned
""",
    "failed_patterns.md": """\
---
type: failed_patterns
last_updated: never
---

# Failed Patterns

Anti-patterns: query/URL combinations that have repeatedly failed.
Listed so the agent can avoid retrying them.
""",
    "user_preferences.md": """\
---
type: user_preferences
last_updated: never
---

# User Preferences

Observations about how the user prefers answers (format, depth, language).
""",
}


# ═══════════════════════════════════════════════════════════════════════
# Memory bundle wrapper (returned from search())
# ═══════════════════════════════════════════════════════════════════════

class MemoryBundle:
    """Single object wrapping the full memory content for one task.

    Implements `format_context()` to be drop-in compatible with the prior
    `MemoryEntry.format_context()` callsite in agent.py.
    """

    __slots__ = ("content",)

    def __init__(self, content: str) -> None:
        self.content = content

    def format_context(self) -> str:
        return self.content


# ═══════════════════════════════════════════════════════════════════════
# Legacy stub — kept only so old imports don't break. Not functional.
# ═══════════════════════════════════════════════════════════════════════

class MemoryEntry:
    """DEPRECATED: legacy JSONL entry shape, retained only for import compat.

    The MD-based MemoryStore no longer produces these. Tests that depend on
    this class must be migrated to the new `MemoryBundle` interface.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "MemoryEntry is deprecated; the MD-based MemoryStore uses "
            "MemoryBundle. See deepresearch/utils/memory_store.py for the "
            "new interface."
        )


# ═══════════════════════════════════════════════════════════════════════
# memory_writer LLM prompt
# ═══════════════════════════════════════════════════════════════════════

_WRITER_SYSTEM_PROMPT = """\
You are a long-term memory writer for a research agent.

Your job: examine the agent's CURRENT MEMORY and the latest task's FINDINGS,
then decide what (if anything) is worth persisting.

CRITICAL RULES:
1. Carefully scan the current memory for OVERLAP with the new findings.
   - If the same fact already exists in memory, prefer NOOP.
   - Only choose REPLACE when the new version is clearly more precise,
     more recent, or more credible than the existing one.
   - Only choose APPEND when the information is genuinely new.
2. Persist GENERAL, REUSABLE knowledge — not task-specific trivia.
   GOOD examples to persist:
     - "GPT-3 arXiv ID = 2005.14165 (Brown et al., 2020)"
     - "arXiv lookups: search '<title> arxiv' returns the abs page first"
     - "arxiv.org: high trust, metadata accuracy near 100%"
   BAD examples (do NOT persist):
     - "User asked about GPT-3 today"
     - "Task hpqa_042 took 12 steps"
     - "Step 3 returned a 500 error"
3. APPEND format:
   - file: target filename only (e.g. "domain_facts.md"), NOT a path.
   - section: target section header (e.g. "## Papers"). If the section is
     not present in the file, the action will create it at the file end.
   - content: the new markdown content (typically a bullet line).
4. REPLACE format:
   - file: target filename only.
   - find: a UNIQUE substring within the file (include enough surrounding
     context so that exactly one match exists). If `find` is ambiguous,
     the action is rejected.
   - replace: the new substring that should occupy the same position.

Output JSON ONLY in this format (no prose, no code fences):
{
  "actions": [
    {"op": "APPEND",  "file": "...", "section": "...", "content": "...", "reason": "..."},
    {"op": "REPLACE", "file": "...", "find": "...", "replace": "...", "reason": "..."},
    {"op": "NOOP",    "reason": "..."}
  ]
}

If nothing is worth persisting from this task, output exactly:
{"actions": []}
"""


# ═══════════════════════════════════════════════════════════════════════
# Main store class
# ═══════════════════════════════════════════════════════════════════════

class MemoryStore:
    """Markdown-based long-term memory with LLM-driven write decisions."""

    DEFAULT_FILES: List[str] = list(_FILE_TEMPLATES.keys())
    INDEX_FILE: str = "MEMORY.md"

    def __init__(
        self,
        store_path: str,
        top_k: int = 3,
        llm_client: Any = None,
        model: str = "qwen-turbo",
    ) -> None:
        """Initialize the markdown memory store.

        Args:
            store_path: Directory path containing the memory MD files.
                If a legacy `.jsonl` file path is passed, the parent
                directory is used (and a sibling dir derived from the
                stem is created).
            top_k: Kept for backward compatibility — no longer affects
                retrieval (the entire memory is always returned).
            llm_client: LLM client used by the memory_writer. Must expose
                `chat(messages, response_format)`. If None, `add()` is a
                no-op (memory becomes read-only).
            model: Model name for the memory_writer call.
        """
        path = Path(store_path)
        # Backward compat: if legacy .jsonl path, use a sibling directory
        if path.suffix == ".jsonl":
            path = path.with_suffix("")
        self.store_path = path
        self.top_k = top_k
        self.llm_client = llm_client
        self.model = model

        self._ensure_initialized()

    # ── Initialization ───────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Create the directory and template files if absent."""
        self.store_path.mkdir(parents=True, exist_ok=True)

        index_file = self.store_path / self.INDEX_FILE
        if not index_file.exists():
            index_file.write_text(_INDEX_TEMPLATE, encoding="utf-8")

        for fname, content in _FILE_TEMPLATES.items():
            f = self.store_path / fname
            if not f.exists():
                f.write_text(content, encoding="utf-8")

    # ── Read path ────────────────────────────────────────────────────

    def _load_all(self) -> str:
        """Concatenate MEMORY.md + all referenced files into a single bundle."""
        sections: List[str] = []

        index_file = self.store_path / self.INDEX_FILE
        if index_file.exists():
            sections.append(
                f"=== {self.INDEX_FILE} ===\n{index_file.read_text(encoding='utf-8')}"
            )

        for fname in self.DEFAULT_FILES:
            f = self.store_path / fname
            if f.exists():
                sections.append(f"=== {fname} ===\n{f.read_text(encoding='utf-8')}")

        return "\n\n".join(sections)

    def search(self, query: str, top_k: Optional[int] = None) -> List[MemoryBundle]:
        """Return the entire memory bundle as a single context.

        For MD-based memory, "retrieval" is in-context: we inject the full
        memory into the consuming LLM's prompt and rely on its attention to
        pick relevant pieces. The `query` and `top_k` parameters are kept
        for interface compatibility but do not affect the returned content.

        Returns an empty list when no category file has user-written entries
        (only templates exist) — avoids polluting the prompt with empty
        scaffolding.
        """
        if not self._has_substantive_content():
            return []
        return [MemoryBundle(self._load_all())]

    def _has_substantive_content(self) -> bool:
        """True iff any category file (not the index) has at least one bullet entry."""
        for fname in self.DEFAULT_FILES:
            f = self.store_path / fname
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                stripped = line.lstrip()
                # Bullet/list/table entries indicate user content beyond template
                if stripped.startswith(("- ", "* ", "+ ", "| ")):
                    return True
        return False

    # ── Write path ───────────────────────────────────────────────────

    async def add(
        self,
        task_id: str,
        query: str,
        summary: str,
        citations: Optional[List[str]] = None,
        key_facts: Optional[List[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """LLM-driven write: decide APPEND/REPLACE/NOOP and apply.

        Returns the list of actions actually applied (or None if skipped
        due to missing LLM client). Never raises — failures are logged.
        """
        if self.llm_client is None:
            logger.info(
                "memory_store: no llm_client configured, skipping add (task_id=%s)",
                task_id,
            )
            return None

        bundle = self._load_all()
        cite_block = "\n".join(f"  - {c}" for c in (citations or [])[:5])
        user_msg = (
            "=== CURRENT MEMORY ===\n"
            f"{bundle}\n\n"
            "=== NEW TASK FINDINGS ===\n"
            f"task_id: {task_id}\n"
            f"query: {query}\n"
            f"summary: {summary[:1500]}\n"
            f"citations:\n{cite_block or '  (none)'}\n\n"
            "What (if anything) should be added or updated in long-term memory?"
        )

        actions = await self._call_writer(user_msg)
        if not actions:
            logger.info(
                "memory_store: writer returned no actions for task=%s", task_id
            )
            return []

        applied: List[Dict[str, Any]] = []
        for action in actions:
            try:
                if self._apply_action(action):
                    applied.append(action)
            except Exception as exc:  # noqa: BLE001 — defensive: never crash agent
                logger.warning(
                    "memory_store: apply action failed (%s) action=%s", exc, action
                )

        if applied:
            logger.info(
                "memory_store: applied %d action(s) for task=%s", len(applied), task_id
            )
        return applied

    async def _call_writer(self, user_msg: str) -> List[Dict[str, Any]]:
        """Call the memory_writer LLM and parse its action list."""
        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": _WRITER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format="json",
            )
            if inspect.isawaitable(response):
                response = await response
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_store: writer call failed (%s)", exc)
            return []

        return self._parse_actions(str(response or ""))

    @staticmethod
    def _parse_actions(raw: str) -> List[Dict[str, Any]]:
        """Parse the writer's response, expecting {'actions': [...]}."""
        text = raw.strip()
        # Strip code fences if the model wrapped JSON in markdown
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("memory_store: invalid JSON from writer (%s)", exc)
            return []
        actions = data.get("actions", [])
        if not isinstance(actions, list):
            return []
        return [a for a in actions if isinstance(a, dict)]

    # ── Action application ──────────────────────────────────────────

    def _apply_action(self, action: Dict[str, Any]) -> bool:
        """Execute one action; return True if it produced a file change."""
        op = str(action.get("op", "")).upper()

        if op == "NOOP":
            return False

        fname = action.get("file", "")
        if not fname or fname not in self.DEFAULT_FILES:
            logger.warning("memory_store: invalid file=%r in action, skipping", fname)
            return False

        target = self.store_path / fname
        if not target.exists():
            logger.warning("memory_store: target file missing %s", target)
            return False

        original = target.read_text(encoding="utf-8")

        if op == "APPEND":
            new_content = self._append_to_section(
                original,
                section=str(action.get("section", "")).strip(),
                content_to_add=str(action.get("content", "")).strip(),
            )
            if new_content == original:
                return False  # no-op append (e.g. empty content)
        elif op == "REPLACE":
            new_content = self._replace_in_file(
                original,
                find=str(action.get("find", "")),
                replace=str(action.get("replace", "")),
            )
            if new_content is None:
                return False  # find was not unique
        else:
            logger.warning("memory_store: unknown op=%r, skipping", op)
            return False

        # Backup before writing
        backup = target.with_suffix(target.suffix + ".bak")
        try:
            shutil.copy2(target, backup)
        except OSError as exc:
            logger.warning("memory_store: backup failed for %s (%s)", target, exc)

        new_content = self._update_frontmatter_timestamp(new_content)
        target.write_text(new_content, encoding="utf-8")
        return True

    @staticmethod
    def _append_to_section(content: str, section: str, content_to_add: str) -> str:
        """Append content into the specified section.

        - If `section` is empty, append at file end.
        - If `section` exists, insert at the end of that section (before the
          next heading of the same or higher level).
        - If `section` does not exist, create it at file end with the content.
        """
        if not content_to_add:
            return content

        if not section:
            return content.rstrip() + "\n\n" + content_to_add + "\n"

        section_norm = section.strip()
        # Make sure section starts with at least one '#'
        if not section_norm.lstrip().startswith("#"):
            section_norm = "## " + section_norm

        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == section_norm:
                level = len(line) - len(line.lstrip("#"))
                end = len(lines)
                for j in range(i + 1, len(lines)):
                    stripped = lines[j].lstrip()
                    if stripped.startswith("#"):
                        next_level = len(lines[j]) - len(stripped)
                        if next_level <= level:
                            end = j
                            break

                # Trim trailing blank lines inside the section
                insert_at = end
                while insert_at > i + 1 and not lines[insert_at - 1].strip():
                    insert_at -= 1

                lines.insert(insert_at, content_to_add)
                result = "\n".join(lines)
                if not result.endswith("\n"):
                    result += "\n"
                return result

        # Section not found — create at end
        return content.rstrip() + f"\n\n{section_norm}\n\n{content_to_add}\n"

    @staticmethod
    def _replace_in_file(content: str, find: str, replace: str) -> Optional[str]:
        """Replace `find` with `replace` only if `find` is unique in the file."""
        if not find:
            return None
        count = content.count(find)
        if count == 0:
            logger.warning("memory_store: REPLACE find string not found")
            return None
        if count > 1:
            logger.warning(
                "memory_store: REPLACE find string not unique (%d matches)", count
            )
            return None
        return content.replace(find, replace, 1)

    @staticmethod
    def _update_frontmatter_timestamp(content: str) -> str:
        """Update `last_updated:` in the YAML frontmatter (if present)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return re.sub(
            r"(last_updated:\s*)([^\n]+)",
            lambda m: m.group(1) + today,
            content,
            count=1,
        )

    # ── Misc ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Approximate footprint: total bytes across memory files."""
        total = 0
        for fname in [self.INDEX_FILE] + self.DEFAULT_FILES:
            f = self.store_path / fname
            if f.exists():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    def clear(self) -> None:
        """Remove all memory + backup files; reinitialize templates."""
        for fname in [self.INDEX_FILE] + self.DEFAULT_FILES:
            target = self.store_path / fname
            if target.exists():
                target.unlink()
            backup = target.with_suffix(target.suffix + ".bak")
            if backup.exists():
                backup.unlink()
        self._ensure_initialized()
