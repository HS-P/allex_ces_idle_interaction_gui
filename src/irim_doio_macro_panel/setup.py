from setuptools import find_packages, setup

package_name = "irim_doio_macro_panel"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/default_mapping.json"]),
        ("share/" + package_name + "/launch", ["launch/doio_macro_panel.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="DOIO macro keyboard -> ROS2 topic publisher (GUI + JSON mapping).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "doio_macro_panel = irim_doio_macro_panel.main:main",
        ],
    },
)
