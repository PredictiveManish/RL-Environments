from setuptools import find_packages, setup

setup(
    name="airport-rl",
    version="1.0.0",
    description="A MuJoCo + Gymnasium RL environment for airport ground operations",
    packages=find_packages(include=["airport_rl", "airport_rl.*"]),
    python_requires=">=3.9",
    install_requires=[
        "mujoco>=3.1.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "train": ["stable-baselines3>=2.3.0", "torch>=2.1.0"],
        "render": ["imageio>=2.31.0", "pillow>=10.0.0"],
        "dev": ["pytest>=7.4.0", "matplotlib>=3.7.0"],
    },
    entry_points={"console_scripts": []},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
