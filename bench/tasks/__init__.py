"""Task registry."""

from bench.task import Task
from bench.tasks import blas, dense, fft, sparse, tensor

ALL_TASKS: list[Task] = [*blas.TASKS, *fft.TASKS, *dense.TASKS, *sparse.TASKS, *tensor.TASKS]
TASKS_BY_ID: dict[str, Task] = {task.id: task for task in ALL_TASKS}

if len(TASKS_BY_ID) != len(ALL_TASKS):
    raise RuntimeError("duplicate task ids")
