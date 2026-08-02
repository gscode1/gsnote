import pytest

from app import db, spaces
from app.prompts import MEMORY
from app.turn import handle_command


def test_timezone_defaults_to_unset():
    assert spaces.get_timezone("u1") is None


def test_timezone_validates_iana_name():
    assert spaces.set_timezone("u1", " Europe/Warsaw ") == "Europe/Warsaw"
    assert spaces.get_timezone("u1") == "Europe/Warsaw"

    with pytest.raises(ValueError, match="Unknown timezone"):
        spaces.set_timezone("u1", "Not/AZone")


def test_timezone_survives_connection_restart():
    spaces.set_timezone("u1", "America/New_York")
    db.get_conn().close()
    db._conn = None
    assert spaces.get_timezone("u1") == "America/New_York"


@pytest.mark.anyio
async def test_timezone_command_reports_sets_and_rejects():
    assert "not set" in (await handle_command("u1", "timezone", "")).lower()
    assert await handle_command("u1", "timezone", "Europe/Warsaw") == "Timezone set to Europe/Warsaw."
    assert await handle_command("u1", "timezone", "") == "Timezone: Europe/Warsaw."
    assert "Unknown timezone" in await handle_command("u1", "timezone", "Not/AZone")


def test_memory_prompt_documents_timezone_lookup():
    assert "get_user_timezone" in MEMORY
    assert "/timezone" in MEMORY
