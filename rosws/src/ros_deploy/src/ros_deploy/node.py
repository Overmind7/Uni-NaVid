#!/usr/bin/env python3
"""ROS node that wraps the UniNaVid agent for online navigation."""

import math
import os
import threading
from typing import Tuple
import yaml

import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo, Image

from ros_deploy.srv import SetInstruction, SetInstructionResponse
from offline_eval_uninavid import UniNaVid_Agent


class UniNaVidRosNode:
    """Subscribe to RGB frames, run the UniNaVid agent, and publish velocity commands."""

    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_camera_info = None
        self.lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._new_command_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._queued_twist = Twist()
        self._queued_duration = 0.0

        self.config = self._load_config()
        self.publish_rate_hz = 10.0
        self.forward_distance_m = 0.5
        self.spin_init_duration_s = 0.5
        self.instruction = rospy.get_param("~instruction", self.config.get("default_instruction", ""))

        rospy.loginfo("Loading UniNaVid agent from %s", self.config["model_path"])
        self.agent = UniNaVid_Agent(self.config["model_path"])

        self.cmd_pub = rospy.Publisher(self.config["cmd_vel_topic"], Twist, queue_size=1)
        self.image_sub = rospy.Subscriber(
            self.config["image_topic"], Image, self._image_callback, queue_size=1, buff_size=2 ** 24
        )
        self.camera_info_sub = rospy.Subscriber(
            self.config["camera_info_topic"], CameraInfo, self._camera_info_callback, queue_size=1
        )

        self.service = rospy.Service("~set_instruction", SetInstruction, self._handle_set_instruction)

        self._publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
        self._publisher_thread.start()
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "UniNaVid ROS node ready. Listening to %s and publishing Twist on %s.",
            self.config["image_topic"],
            self.config["cmd_vel_topic"],
        )

    def _load_config(self):
        pkg_path = rospkg.RosPack().get_path("ros_deploy")
        default_path = os.path.join(pkg_path, "config", "default.yaml")
        config_path = rospy.get_param("~config_path", default_path)
        config_path = os.path.realpath(config_path)

        if not os.path.exists(config_path):
            raise rospy.ROSException(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        required_keys = ["image_topic", "camera_info_topic", "cmd_vel_topic", "model_path"]
        for key in required_keys:
            if key not in config:
                raise rospy.ROSException(f"Missing required key '{key}' in config {config_path}")

        config.setdefault("linear_speed", 0.25)
        config.setdefault("angular_speed", 0.5)
        config.setdefault("default_instruction", "")
        return config

    def _handle_set_instruction(self, req: SetInstruction.Request) -> SetInstructionResponse:
        instruction = req.instruction.strip()
        with self.lock:
            self.instruction = instruction
            rospy.set_param("~instruction", instruction)
        rospy.loginfo("Instruction updated via service: %s", instruction)
        return SetInstructionResponse(success=True, message="Instruction updated")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        with self.lock:
            self.latest_camera_info = msg

    def _image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            rospy.logerr("Failed to convert image: %s", exc)
            return

        with self.lock:
            self.latest_image = np.asarray(cv_image, dtype=np.uint8)
            camera_info = self.latest_camera_info
            instruction = rospy.get_param("~instruction", self.instruction)
            if instruction != self.instruction:
                rospy.loginfo("Instruction updated from parameter server: %s", instruction)
                self.instruction = instruction

        if camera_info is None:
            rospy.logwarn_throttle(30.0, "Waiting for camera info on %s", self.config["camera_info_topic"])

        self._process_frame()

    def _process_frame(self) -> None:
        with self.lock:
            if self.latest_image is None:
                return
            rgb_np = self.latest_image.copy()
            instruction = self.instruction

        if not instruction:
            rospy.logwarn_throttle(30.0, "Instruction is empty; agent will run with blank prompt.")

        action_data = self.agent.act({"observations": rgb_np, "instruction": instruction})
        actions = action_data.get("actions", []) if isinstance(action_data, dict) else []
        if not actions:
            rospy.logwarn("No actions returned by UniNaVid agent; publishing stop Twist.")
            twist = Twist()
            self.cmd_pub.publish(twist)
            return

        primary_action = actions[0].strip().lower()
        twist, duration = self._action_to_twist(primary_action)
        rospy.loginfo(
            "Action %s -> Twist linear.x=%.3f angular.z=%.3f for %.2f seconds @ %.1f Hz",
            primary_action,
            twist.linear.x,
            twist.angular.z,
            duration,
            self.publish_rate_hz,
        )
        self._queue_command(twist, duration)

    def _action_to_twist(self, action: str) -> Tuple[Twist, float]:
        twist = Twist()
        duration = 0.0
        if action == "forward":
            twist.linear.x = float(self.config["linear_speed"])
            if twist.linear.x > 0:
                duration = self.forward_distance_m / twist.linear.x
        elif action == "left":
            twist.angular.z = float(self.config["angular_speed"])
            if twist.angular.z != 0:
                duration = math.radians(30.0) / abs(twist.angular.z) + self.spin_init_duration_s
        elif action == "right":
            twist.angular.z = -float(self.config["angular_speed"])
            if twist.angular.z != 0:
                duration = math.radians(30.0) / abs(twist.angular.z) + self.spin_init_duration_s
        elif action == "stop":
            pass
        else:
            rospy.logwarn("Unknown action '%s'; sending zero Twist.", action)
        return twist, duration

    def _queue_command(self, twist: Twist, duration: float) -> None:
        """Send a command to the publisher thread, preempting any existing one."""

        with self._command_lock:
            self._queued_twist = twist
            self._queued_duration = max(duration, 0.0)
        self._new_command_event.set()

    def _publisher_loop(self) -> None:
        rate = rospy.Rate(self.publish_rate_hz)
        while not self._shutdown_event.is_set():
            self._new_command_event.wait()
            if self._shutdown_event.is_set():
                break

            self._new_command_event.clear()
            with self._command_lock:
                twist = self._queued_twist
                duration = self._queued_duration

            start_time = rospy.Time.now().to_sec()
            end_time = start_time + duration

            while not self._shutdown_event.is_set():
                if self._new_command_event.is_set():
                    self.cmd_pub.publish(Twist())
                    break

                self.cmd_pub.publish(twist)

                if duration <= 0:
                    break

                if rospy.Time.now().to_sec() >= end_time:
                    self.cmd_pub.publish(Twist())
                    break

                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    break

    def _on_shutdown(self) -> None:
        self._shutdown_event.set()
        self._new_command_event.set()
        try:
            self._publisher_thread.join(timeout=1.0)
        except Exception:  # noqa: BLE001
            pass
        self.cmd_pub.publish(Twist())


def main() -> None:
    rospy.init_node("uninavid_agent")
    UniNaVidRosNode()
    rospy.spin()


if __name__ == "__main__":
    main()
