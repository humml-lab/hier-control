from .value_scheduler import ScheduleConfig, ValueScheduler


class MultiValueScheduler:
    def __init__(self, value_schedules: dict[str, ScheduleConfig], n_epochs: int):
        self.schedulers = {
            name: ValueScheduler(cfg, n_epochs) for name, cfg in value_schedules.items()
        }

    def step(self, epoch: int):
        for name, scheduler in self.schedulers.items():
            scheduler.step(epoch)

    def get(self, name: str, largest_possible: bool = False):
        if name not in self.schedulers:
            raise ValueError(f"Scheduler for '{name}' not found.")
        if largest_possible:
            return self.schedulers[name].largest_value
        return self.schedulers[name].value

    def is_used(self, name: str):
        if name not in self.schedulers:
            raise ValueError(f"Scheduler for '{name}' not found.")
        return self.schedulers[name].is_used
