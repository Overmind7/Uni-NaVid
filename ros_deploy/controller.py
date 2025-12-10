"""Translate symbolic navigation commands into timed Twist instructions.

This module keeps dependencies light by defining a small Twist-like dataclass
instead of importing ROS message types. It provides convenience helpers for
planning sequences with optional ramp-up/down smoothing and a cooperative stop
flag to allow preemption.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List, Sequence
import threading


@dataclass
class TwistCommand:
    """Simple Twist-like command representation with a duration.

    Attributes:
        linear_x: Forward (+) or backward (-) linear velocity in m/s.
        angular_z: Angular velocity around the Z axis in rad/s. Positive
            values rotate left, negative values rotate right.
        duration: How long to apply the command in seconds.
    """

    linear_x: float = 0.0
    angular_z: float = 0.0
    duration: float = 0.0

    def scale(self, factor: float) -> "TwistCommand":
        """Return a scaled copy of this command leaving duration unchanged."""
        return TwistCommand(
            linear_x=self.linear_x * factor,
            angular_z=self.angular_z * factor,
            duration=self.duration,
        )


class MotionController:
    """Convert a list of directions into timed Twist commands."""

    def __init__(
        self,
        linear_speed: float = 0.2,
        angular_speed: float = 0.4,
        step_duration: float = 1.0,
        ramp_duration: float = 0.2,
        ramp_steps: int = 5,
    ) -> None:
        """Initialize controller with timing and speed defaults.

        Args:
            linear_speed: Magnitude of forward/backward speed in m/s.
            angular_speed: Magnitude of turning speed in rad/s.
            step_duration: Base duration for each direction command.
            ramp_duration: Total time spent ramping up and down.
            ramp_steps: Number of interpolation steps per ramp segment.
        """
        if ramp_steps < 1:
            raise ValueError("ramp_steps must be >= 1")
        if step_duration <= 0:
            raise ValueError("step_duration must be positive")
        if ramp_duration < 0:
            raise ValueError("ramp_duration must be non-negative")

        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.step_duration = step_duration
        self.ramp_duration = ramp_duration
        self.ramp_steps = ramp_steps
        self._stop_event = threading.Event()

    @property
    def stop_requested(self) -> bool:
        """Return True if a stop has been requested."""

        return self._stop_event.is_set()

    def request_stop(self) -> None:
        """Signal that planning should halt early."""

        self._stop_event.set()

    def reset_stop(self) -> None:
        """Clear any previously set stop signal."""

        self._stop_event.clear()

    def plan_commands(self, directions: Iterable[str]) -> List[TwistCommand]:
        """Plan a full sequence of Twist commands from symbolic directions.

        The returned list includes ramp-up and ramp-down segments to smooth
        velocity changes and reduce jerk.
        """

        planned: List[TwistCommand] = []
        for direction in directions:
            if self.stop_requested:
                break
            base = self._direction_to_twist(direction)
            planned.extend(self._with_ramp(base))
        return planned

    def _direction_to_twist(self, direction: str) -> TwistCommand:
        normalized = direction.strip().lower()
        if normalized == "forward":
            return TwistCommand(linear_x=self.linear_speed, angular_z=0.0, duration=self.step_duration)
        if normalized == "backward":
            return TwistCommand(linear_x=-self.linear_speed, angular_z=0.0, duration=self.step_duration)
        if normalized == "left":
            return TwistCommand(linear_x=0.0, angular_z=self.angular_speed, duration=self.step_duration)
        if normalized == "right":
            return TwistCommand(linear_x=0.0, angular_z=-self.angular_speed, duration=self.step_duration)
        raise ValueError(f"Unknown direction: {direction}")

    def _with_ramp(self, base: TwistCommand) -> List[TwistCommand]:
        """Return ramped commands (up, constant, down) for a base command."""

        # If no ramping requested or no motion, return base as-is.
        if self.ramp_duration <= 0 or (base.linear_x == 0 and base.angular_z == 0):
            return [base]

        ramp_time = min(self.ramp_duration, base.duration / 2)
        plateau_time = max(base.duration - 2 * ramp_time, 0)
        per_step_duration = ramp_time / self.ramp_steps if ramp_time > 0 else 0

        segments: List[TwistCommand] = []
        if ramp_time > 0:
            # Ramp up from 0 to 1.0 multiplier
            for step in range(1, self.ramp_steps + 1):
                if self.stop_requested:
                    return segments
                factor = step / self.ramp_steps
                segments.append(
                    TwistCommand(
                        linear_x=base.linear_x * factor,
                        angular_z=base.angular_z * factor,
                        duration=per_step_duration,
                    )
                )

        if plateau_time > 0 and not self.stop_requested:
            segments.append(
                TwistCommand(
                    linear_x=base.linear_x,
                    angular_z=base.angular_z,
                    duration=plateau_time,
                )
            )

        if ramp_time > 0 and not self.stop_requested:
            for step in reversed(range(1, self.ramp_steps + 1)):
                if self.stop_requested:
                    return segments
                factor = step / self.ramp_steps
                segments.append(
                    TwistCommand(
                        linear_x=base.linear_x * factor,
                        angular_z=base.angular_z * factor,
                        duration=per_step_duration,
                    )
                )

        return segments


def describe_command_plan(commands: Sequence[TwistCommand]) -> List[str]:
    """Return formatted descriptions for a sequence of commands."""

    descriptions: List[str] = []
    cumulative_time = 0.0
    for cmd in commands:
        cumulative_time += cmd.duration
        descriptions.append(
            f"t={cumulative_time:5.2f}s | lin_x={cmd.linear_x:+.3f} m/s | "
            f"ang_z={cmd.angular_z:+.3f} rad/s | dt={cmd.duration:.2f}s"
        )
    return descriptions


def dry_run_commands(
    directions: Iterable[str],
    *,
    linear_speed: float = 0.2,
    angular_speed: float = 0.4,
    step_duration: float = 1.0,
    ramp_duration: float = 0.2,
    ramp_steps: int = 5,
    logger: logging.Logger | None = None,
) -> List[TwistCommand]:
    """Plan commands and emit log output for debugging.

    Returns the planned commands to allow unit-style assertions without
    requiring hardware.
    """

    controller = MotionController(
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        step_duration=step_duration,
        ramp_duration=ramp_duration,
        ramp_steps=ramp_steps,
    )
    commands = controller.plan_commands(directions)

    active_logger = logger or logging.getLogger(__name__)
    for line in describe_command_plan(commands):
        active_logger.info(line)

    return commands
