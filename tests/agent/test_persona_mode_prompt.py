"""Regression tests for persona-mode prompt assembly (Phase 1, Telegram).

Two failures fixed here:
  * HERMES_HOME/AGENTS.md was never discovered on gateway persona sessions
    (their cwd resolves to Path.home() via the terminal.cwd placeholder), so
    the texting-voice / media / register-mirroring rules never reached the
    stored system prompt.
  * the engine's engineering discipline blocks (mandatory_tool_use /
    act_dont_ask / verification) shipped into the persona chat.
"""

from __future__ import annotations

from pathlib import Path

from agent.prompt_builder import build_context_files_prompt
from agent import system_prompt as sp


def test_persona_home_agents_md_loads_into_prompt(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "AGENTS.md").write_text(
        "# Persona rules\n\n## Texting voice\nshort, lowercase, punchy\n"
        "\n## Image & GIF sending\nmedia is context-based\n",
        encoding="utf-8",
    )
    work = tmp_path / "workspace"
    work.mkdir()

    out = build_context_files_prompt(cwd=str(work), skip_soul=True, persona_home=home)

    assert "## AGENTS.md" in out
    assert "Texting voice" in out
    assert "Image & GIF sending" in out


def test_persona_home_absent_does_not_inject(tmp_path):
    work = tmp_path / "workspace"
    work.mkdir()
    out = build_context_files_prompt(cwd=str(work), skip_soul=True)
    assert "Texting voice" not in out


def test_persona_home_leads_project_context(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "AGENTS.md").write_text("# Persona\npersona marker\n", encoding="utf-8")
    work = tmp_path / "workspace"
    work.mkdir()
    (work / "AGENTS.md").write_text("# Project\nproject marker\n", encoding="utf-8")

    out = build_context_files_prompt(cwd=str(work), skip_soul=True, persona_home=home)

    assert "persona marker" in out and "project marker" in out
    assert out.index("persona marker") < out.index("project marker")


class _FakeAgent:
    def __init__(self, platform: str, persona_mode: bool = True):
        self.platform = platform
        self._persona_mode = persona_mode


def test_persona_mode_active_only_on_chat_surfaces():
    assert sp._persona_mode_active(_FakeAgent("telegram")) is True
    assert sp._persona_mode_active(_FakeAgent("discord")) is True
    assert sp._persona_mode_active(_FakeAgent("cli")) is False
    assert sp._persona_mode_active(_FakeAgent("tui")) is False
    assert sp._persona_mode_active(_FakeAgent("telegram", persona_mode=False)) is False


def test_persona_mode_skips_discipline_blocks():
    agent = _FakeAgent("telegram")
    agent.valid_tool_names = {"terminal", "web_search"}
    agent.model = "deepseek-v4-flash"
    agent._tool_use_enforcement = "auto"
    agent._execution_guidance = "auto"
    agent._task_completion_guidance = True
    agent._parallel_tool_call_guidance = True

    persona_joined = "\n".join(p for p in sp._guidance_parts(agent) if p)
    assert "mandatory_tool_use" not in persona_joined
    assert "act_dont_ask" not in persona_joined

    agent.platform = "cli"
    coding_joined = "\n".join(p for p in sp._guidance_parts(agent) if p)
    assert "mandatory_tool_use" in coding_joined


def test_agent_init_reads_persona_mode():
    from agent.agent_init import _apply_agent_section

    class _A:
        run_budget_seconds = None

    a = _A()
    _apply_agent_section(a, {"agent": {"persona_mode": True}})
    assert a._persona_mode is True
    b = _A()
    _apply_agent_section(b, {"agent": {}})
    assert b._persona_mode is False
