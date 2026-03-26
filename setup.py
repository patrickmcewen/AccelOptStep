from setuptools import setup

setup(
    name="accelopt",
    version="0.0.1",
    packages=["accelopt"],
    install_requires=[
        "openai-agents>=0.13",
        "openai>=2.0",
        "logfire",
        "pandas",
        "pydantic",
        "torch",
        "sympy",
        "networkx",
        "matplotlib",
        "numpy",
        "aiohttp",
    ],
)
