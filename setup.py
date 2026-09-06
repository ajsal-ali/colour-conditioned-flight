"""gym-mujoco-drones package setup."""

from setuptools import setup, find_packages

setup(
    name="gym-mujoco-drones",
    version="1.0.0",
    description="MuJoCo-based multi-drone Gymnasium environments for RL",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.29.0",
        "mujoco>=3.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "rl": ["stable-baselines3>=2.0.0"],
        "marl": ["pettingzoo>=1.24.0"],
        "viz": ["matplotlib>=3.5.0"],
        "all": [
            "stable-baselines3>=2.0.0",
            "pettingzoo>=1.24.0",
            "matplotlib>=3.5.0",
            "Pillow>=9.0.0",
        ],
    },
)
