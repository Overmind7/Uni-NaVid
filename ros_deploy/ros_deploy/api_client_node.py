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


class UniNaVidApiRosNode:
    """Subscribe to RGB frames, call remote Uni-NaVid API, and publish velocity."""

    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        self.config = self._load_config()
        self.server_url = rospy.get_param("~server_url", self.config["server_url"]).rstrip("/")
        self.instruction = rospy.get_param("~instruction", self.config.get("default_instruction", ""))
        self.request_timeout = float(self.config.get("request_timeout", 5.0))
        self.session = requests.Session()

        self.cmd_pub = rospy.Publisher(self.config["cmd_vel_topic"], Twist, queue_size=1)
        self.image_sub = rospy.Subscriber(
            self.config["image_topic"], Image, self._image_callback, queue_size=1, buff_size=2 ** 24
        )

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

        if not instruction:
            rospy.logwarn_throttle(30.0, "Instruction is empty; API call will use a blank prompt.")

        actions = self._query_server(rgb_np, instruction)
        if not actions:
            rospy.logwarn("No actions returned by Uni-NaVid API; publishing stop Twist.")
            twist = Twist()
        else:
            primary_action = actions[0].strip().lower()
            twist = self._action_to_twist(primary_action)
            rospy.loginfo(
                "Action %s -> Twist linear.x=%.3f angular.z=%.3f",
                primary_action,
                twist.linear.x,
                twist.angular.z,
            )

        self.cmd_pub.publish(twist)

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

    def _action_to_twist(self, action: str) -> Twist:
        twist = Twist()
        if action == "forward":
            twist.linear.x = float(self.config["linear_speed"])
        elif action == "left":
            twist.angular.z = float(self.config["angular_speed"])
        elif action == "right":
            twist.angular.z = -float(self.config["angular_speed"])
        elif action == "backward":
            twist.linear.x = -float(self.config["linear_speed"])
        elif action == "stop":
            pass
        else:
            rospy.logwarn("Unknown action '%s'; sending zero Twist.", action)
        return twist


def main() -> None:
    rospy.init_node("uninavid_api_agent")
    UniNaVidApiRosNode()
    rospy.spin()


if __name__ == "__main__":
    main()
