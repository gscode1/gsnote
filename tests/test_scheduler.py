from app import scheduler


def test_scheduler_uses_one_minute_worker_for_reminders_and_briefings(monkeypatch):
    class FakeScheduler:
        def __init__(self):
            self.jobs = []

        def add_job(self, fn, trigger, **kwargs):
            self.jobs.append(kwargs["id"])

        def start(self):
            pass

        def shutdown(self, wait=False):
            pass

    fake = FakeScheduler()
    monkeypatch.setattr(scheduler, "AsyncIOScheduler", lambda: fake)
    scheduler.start_scheduler(lambda _: None, lambda *_: None)

    assert "reminders_tick" in fake.jobs
    assert "daily_briefing" not in fake.jobs
    scheduler.stop_scheduler()
