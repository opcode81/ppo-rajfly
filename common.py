from collections import deque
from collections.abc import Callable

import numpy as np
import pandas as pd
from tianshou.utils.logger.base import VALID_LOG_VALS_TYPE, BaseLogger, DataScope


class TrainLogger(BaseLogger):
    """A logger that stores global step and running mean reward (train)."""

    def __init__(self, train_interval) -> None:
        super().__init__(train_interval, 0, 0, 0)
        self.progress_data = {"global_step": [], "mean_reward": []}
        self.reward_buffer = deque(maxlen=100)

    def log_train_data(self, log_data: dict, step: int) -> None:
        """Log step and mean reward.

        :param log_data: a dict containing the information returned by the collector during the train step.
        :param step: stands for the timestep the collector result is logged.
        """
        if step - self.last_log_train_step >= self.train_interval:
            self.reward_buffer.extend(log_data["returns"])
            mean_reward = (
                np.nan
                if len(self.reward_buffer) == 0
                else float(np.mean(self.reward_buffer))
            )
            self.progress_data["global_step"].append(step)
            self.progress_data["mean_reward"].append(mean_reward)
            self.last_log_train_step = step

    def write(
        self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]
    ) -> None:
        pass

    def log_test_data(self, log_data: dict, step: int) -> None:
        pass

    def log_update_data(self, log_data: dict, step: int) -> None:
        pass

    def log_info_data(self, log_data: dict, step: int) -> None:
        pass

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        pass

    def restore_data(self) -> tuple[int, int, int]:
        return 0, 0, 0
