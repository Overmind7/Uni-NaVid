"""Compatibility wrapper for ros_deploy.api_server entrypoint."""

from ros_deploy.ros_deploy.api_server import *  # noqa: F401,F403

if __name__ == "__main__":
    from ros_deploy.ros_deploy.api_server import main

    main()
