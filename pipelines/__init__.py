"""The Pipelines module is created to facilitate the use of the library. It provides ways to
perform end-to-end operation such as model training or generation. A typical Pipeline is composed by
several disco's instances which are articulated together.

A :class:`__call__` function is defined and used to launch the Pipeline. """

from .disco_training_config import discoTrainingConfig
from .base_pipeline import Pipeline
from .generation import TimeGenerationPipeline
from .training import TrainingPipeline

__all__ = ["Pipeline", "TrainingPipeline", "TimeGenerationPipeline"]
