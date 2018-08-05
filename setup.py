from distutils.core import setup

setup(
    name = "timspyutils",
    version = "0.1dev",
    packages = [
        "timspyutils",
    ],
    install_require = [
        "numpy>1.14.0",
    ],
    tests_require = [
        "pytest>3.5.1",
    ],
    license = "MIT",
    long_description = open("README.md").read(),
)