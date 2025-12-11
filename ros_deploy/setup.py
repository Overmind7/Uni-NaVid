from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=["ros_deploy"],
    package_dir={"": ""},
)

setup(**setup_args)
