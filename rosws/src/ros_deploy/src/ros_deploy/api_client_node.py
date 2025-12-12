#!/usr/bin/env python3
"""ROS node that proxies Uni-NaVid API responses into Twist commands."""

import base64
import os
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests
import rospkg
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import rospy
from sensor_msgs.msg import Image

from ros_deploy import MotionController, TwistCommand, describe_command_plan


class UniNaVidApiRosNode:
    """Subscribe to RGB frames, call remote Uni-NaVid API, and publish velocity."""

    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._new_command_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._queued_commands: List[TwistCommand] = []
        self._active_actions: Optional[List[str]] = None
        self._commands_in_progress = False
        self.last_request_time = rospy.Time(0)

        self.config = self._load_config()
        self.server_url = rospy.get_param("~server_url", self.config["server_url"]).rstrip("/")
        self.instruction = rospy.get_param("~instruction", self.config.get("default_instruction", ""))
        self.request_timeout = float(self.config.get("request_timeout", 5.0))
        self.session = requests.Session()

        self.cmd_pub = rospy.Publisher(self.config["cmd_vel_topic"], Twist, queue_size=1)
        self.image_sub = rospy.Subscriber(
            self.config["image_topic"], Image, self._image_callback, queue_size=1, buff_size=2 ** 24
        )

        self.linear_speed = float(rospy.get_param("~linear_speed", self.config["linear_speed"]))
        self.angular_speed = float(rospy.get_param("~angular_speed", self.config["angular_speed"]))
        self.ramp_duration = float(rospy.get_param("~ramp_duration", self.config["ramp_duration"]))
        self.ramp_steps = int(rospy.get_param("~ramp_steps", self.config["ramp_steps"]))
        self.forward_distance_m = 0.5
        self.spin_angle_rad = float(rospy.get_param("~spin_angle_rad", 3.141592653589793 / 6))
        self.spin_init_duration_s = float(rospy.get_param("~spin_init_duration_s", 0.5))
        self.publish_rate_hz = 10.0
        self.api_cooldown_s = float(rospy.get_param("~api_cooldown_s", 0.5))

        self.controller = MotionController(
            linear_speed=self.linear_speed,
            angular_speed=self.angular_speed,
            forward_distance_m=self.forward_distance_m,
            spin_angle_rad=self.spin_angle_rad,
            spin_init_duration_s=self.spin_init_duration_s,
            ramp_duration=self.ramp_duration,
            ramp_steps=self.ramp_steps,
        )

        self._publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
        self._publisher_thread.start()
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "Uni-NaVid API ROS node ready. Sending requests to %s and publishing Twist on %s.",
            self.server_url,
            self.config["cmd_vel_topic"],
        )

        self._warmup_server()

    def _load_config(self) -> Dict[str, object]:
        pkg_path = rospkg.RosPack().get_path("ros_deploy")
        default_path = os.path.join(pkg_path, "config", "api_client.yaml")
        config_path = rospy.get_param("~config_path", default_path)
        config_path = os.path.realpath(config_path)

        if not os.path.exists(config_path):
            raise rospy.ROSException(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        required_keys = ["image_topic", "cmd_vel_topic", "server_url"]
        for key in required_keys:
            if key not in config:
                raise rospy.ROSException(f"Missing required key '{key}' in config {config_path}")

        config.setdefault("linear_speed", 0.25)
        config.setdefault("angular_speed", 0.5)
        config.setdefault("ramp_duration", 0.2)
        config.setdefault("ramp_steps", 5)
        config.setdefault("default_instruction", "")
        config.setdefault("request_timeout", 5.0)
        return config

    def _warmup_server(self) -> None:
        warmup_url = f"{self.server_url}/warmup"
        try:
            response = self.session.post(warmup_url, timeout=self.request_timeout)
            response.raise_for_status()
            rospy.loginfo("Warmup response: %s", response.json())
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn("Failed to warm up Uni-NaVid API at %s: %s", warmup_url, exc)

    def _image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            rospy.logerr("Failed to convert image: %s", exc)
            return

        with self.lock:
            self.latest_image = cv_image
            instruction = rospy.get_param("~instruction", self.instruction)
            if instruction != self.instruction:
                rospy.loginfo("Instruction updated from parameter server: %s", instruction)
                self.instruction = instruction

        self._process_frame()

    def _process_frame(self) -> None:
        with self.lock:
            if self.latest_image is None:
                return
            rgb_np = self.latest_image.copy()
            instruction = self.instruction

        now = rospy.Time.now()
        if (now - self.last_request_time).to_sec() < self.api_cooldown_s:
            rospy.logdebug_throttle(
                5.0, "Within API cooldown window; skipping this frame's request."
            )
            return
        self.last_request_time = now

        if not instruction:
            rospy.logwarn_throttle(30.0, "Instruction is empty; API call will use a blank prompt.")

        actions = self._query_server(rgb_np, instruction)
        self._execute_actions(actions)

    def _query_server(self, bgr_image, instruction: str) -> List[str]:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode(".jpg", rgb_image)
        if not success:
            rospy.logerr("Failed to encode image for Uni-NaVid API request.")
            return []

        image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        payload = {"instruction": instruction, "image_base64": image_base64}
        predict_url = f"{self.server_url}/predict"
        try:
            response = self.session.post(predict_url, json=payload, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
            actions = data.get("actions", []) if isinstance(data, dict) else []
            if not isinstance(actions, list):
                rospy.logwarn("Unexpected response format from Uni-NaVid API: %s", data)
                return []
            return [str(action) for action in actions]
        except Exception as exc:  # noqa: BLE001
            rospy.logerr("Request to Uni-NaVid API failed: %s", exc)
            return []

    def _execute_actions(self, actions: List[str]) -> None:
        if not actions:
            rospy.logwarn("No actions returned by Uni-NaVid API; publishing stop Twist.")
            self._stop_active_commands()
            return

        normalized_actions = [str(action).strip().lower() for action in actions if str(action).strip()]
        if not normalized_actions:
            rospy.logwarn("Actions list was empty after normalization; publishing stop Twist.")
            self._stop_active_commands()
            return

        with self._command_lock:
            if self._commands_in_progress and self._active_actions == normalized_actions:
                rospy.loginfo_throttle(
                    10.0, "Received same actions while commands are in progress; ignoring update."
                )
                return

        if "stop" in normalized_actions:
            rospy.loginfo("Stop action received; requesting controller stop and publishing zero velocity.")
            self._stop_active_commands()
            return

        self._stop_active_commands(publish_stop=False)
        self.controller.reset_stop()
        try:
            commands = self.controller.plan_commands(normalized_actions)
        except ValueError as exc:
            rospy.logwarn("Failed to plan commands from actions %s: %s", normalized_actions, exc)
            self._stop_active_commands()
            return

        if not commands:
            rospy.logwarn("Planning produced no commands; publishing stop Twist.")
            self._stop_active_commands()
            return

        rospy.loginfo("Planned %d Twist commands:", len(commands))
        for line in describe_command_plan(commands):
            rospy.loginfo(line)

        self._queue_commands(commands, normalized_actions)

    def _queue_commands(self, commands: List[TwistCommand], normalized_actions: List[str]) -> None:
        with self._command_lock:
            self._queued_commands = list(commands)
            self._active_actions = list(normalized_actions)
            self._commands_in_progress = True
        self._new_command_event.set()

    def _stop_active_commands(self, *, publish_stop: bool = True) -> None:
        self.controller.request_stop()
        with self._command_lock:
            self._queued_commands = []
            self._commands_in_progress = False
            self._active_actions = None
        self._new_command_event.set()
        self.controller.reset_stop()
        if publish_stop:
            self._publish_stop_twist()

    def _publisher_loop(self) -> None:
        rate = rospy.Rate(self.publish_rate_hz)
        while not self._shutdown_event.is_set():
            self._new_command_event.wait()
            if self._shutdown_event.is_set():
                break

            self._new_command_event.clear()
            with self._command_lock:
                commands = list(self._queued_commands)
                self._queued_commands = []
                active_actions = self._active_actions

            if not commands:
                self._publish_stop_twist()
                with self._command_lock:
                    self._commands_in_progress = False
                    self._active_actions = None
                continue

            rospy.loginfo("Executing %d commands derived from actions: %s", len(commands), active_actions)
            self.controller.reset_stop()
            preempted = False

            for command in commands:
                if command.duration <= 0:
                    continue

                twist = Twist()
                twist.linear.x = command.linear_x
                twist.angular.z = command.angular_z

                end_time = rospy.Time.now() + rospy.Duration.from_sec(command.duration)
                while rospy.Time.now() < end_time and not rospy.is_shutdown():
                    if (
                        self._shutdown_event.is_set()
                        or self._new_command_event.is_set()
                        or self.controller.stop_requested
                    ):
                        preempted = True
                        break
                    self.cmd_pub.publish(twist)
                    try:
                        rate.sleep()
                    except rospy.ROSInterruptException:
                        preempted = True
                        break

                if preempted or self._shutdown_event.is_set():
                    break

            if not preempted:
                self._publish_stop_twist()

            with self._command_lock:
                if not preempted:
                    self._commands_in_progress = False
                    self._active_actions = None

    def _on_shutdown(self) -> None:
        self._shutdown_event.set()
        self._new_command_event.set()
        try:
            self._publisher_thread.join(timeout=1.0)
        except Exception:  # noqa: BLE001
            pass

    def _publish_stop_twist(self) -> None:
        twist = Twist()
        self.cmd_pub.publish(twist)


def main() -> None:
    rospy.init_node("uninavid_api_agent")
    UniNaVidApiRosNode()
    rospy.spin()


if __name__ == "__main__":
    main()
