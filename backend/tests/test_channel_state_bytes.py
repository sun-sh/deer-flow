"""IM channel JSON state must be UTF-8 with LF on every platform.

These stores persist small JSON documents through text mode. A default code page
can make non-ASCII ids unreadable, and default newline translation turns LF into
CRLF on Windows, so the on-disk bytes differ from the serialized document even
though ``json.loads`` still succeeds.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.channels.buzz_seen_events import BuzzSeenEventStore
from app.channels.discord import DiscordChannel
from app.channels.message_bus import MessageBus
from app.channels.runtime_config_store import ChannelRuntimeConfigStore
from app.channels.store import JsonChannelStore
from app.channels.wechat import WechatChannel


class _FakeChannelStore:
    def __init__(self, tmp_path: Path) -> None:
        self.channels_dir = tmp_path


def _assert_utf8_lf(raw: bytes) -> None:
    assert b"\r\n" not in raw
    raw.decode("utf-8")


@pytest.mark.asyncio
async def test_discord_thread_map_is_utf8_lf(tmp_path: Path) -> None:
    channel = DiscordChannel(bus=MessageBus(), config={"bot_token": "t", "channel_store": _FakeChannelStore(tmp_path)})
    channel._record_thread_mapping("chan-\u8d44\u6e90", "thread-\u8ba8\u8bba")

    await asyncio.to_thread(channel._persist_thread_mappings)

    _assert_utf8_lf((tmp_path / "discord_threads.json").read_bytes())


def test_json_channel_store_is_utf8_lf(tmp_path: Path) -> None:
    path = tmp_path / "channels" / "store.json"
    store = JsonChannelStore(path)
    store._data = {"slack:C-\u8d44\u6e90": {"thread_id": "t-\u8ba8\u8bba", "user_id": "\u7528\u6237", "created_at": 1.0, "updated_at": 1.0}}

    store._save()

    _assert_utf8_lf(path.read_bytes())


def test_channel_runtime_config_is_utf8_lf(tmp_path: Path) -> None:
    path = tmp_path / "channels" / "runtime-config.json"
    store = ChannelRuntimeConfigStore(path)

    store.set_provider_config("buzz", {"enabled": True, "relay_url": "wss://relay.example/\u804a\u5929"})

    _assert_utf8_lf(path.read_bytes())


def test_buzz_seen_events_are_utf8_lf(tmp_path: Path) -> None:
    path = tmp_path / "channels" / "buzz-seen-events.json"
    store = BuzzSeenEventStore(path)
    store.record("channel-\u8d44\u6e90", "event-\u8ba8\u8bba")
    store.flush()

    _assert_utf8_lf(path.read_bytes())


def test_wechat_cursor_and_auth_are_utf8_lf(tmp_path: Path) -> None:
    channel = WechatChannel(MessageBus(), config={"bot_token": "token-\u8ba8\u8bba", "state_dir": str(tmp_path)})
    channel._get_updates_buf = "cursor-\u8ba8\u8bba"

    channel._save_state()
    channel._save_auth_state(status="confirmed")

    _assert_utf8_lf((tmp_path / "wechat-getupdates.json").read_bytes())
    _assert_utf8_lf((tmp_path / "wechat-auth.json").read_bytes())
