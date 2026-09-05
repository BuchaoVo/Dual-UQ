"""Explicit model capabilities."""

from enum import Enum


class ModelCapability(str, Enum):
    LOCAL_SCORING = "LOCAL_SCORING"
    GENERATION = "GENERATION"
    SEQUENCE_SCORING = "SEQUENCE_SCORING"
    MULTISTATE_GENERATION = "MULTISTATE_GENERATION"
