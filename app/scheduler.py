"""In-process APScheduler — portable across docker-compose + k8s (PRD §12)."""
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_settings
from app.reminders import run_reminders
from app.resurfacing import run_digest

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


def start_scheduler(send_fn, reminder_send_fn) -> AsyncIOScheduler:
    global _scheduler
    settings = get_settings()
    _scheduler = AsyncIOScheduler()

    if settings.resurfacing_enabled:
        trigger = CronTrigger.from_crontab(settings.resurfacing_cron)

        async def _job():
            try:
                result = await run_digest(send_fn)
                logger.info("resurfacing digest run: %s", result)
            except Exception:
                logger.exception("resurfacing digest job failed")

        _scheduler.add_job(_job, trigger, id="resurfacing_digest")

    async def _reminders_job():
        try:
            result = await run_reminders(reminder_send_fn)
            logger.info("reminder run: %s", result)
        except Exception:
            logger.exception("reminder job failed")

    _scheduler.add_job(
        _reminders_job, CronTrigger.from_crontab(settings.reminder_cron), id="reminders_tick"
    )

    _scheduler.start()
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
