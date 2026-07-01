#!/usr/bin/env python3
"""Setup script for ChronicleMem."""

from setuptools import find_packages, setup
import os


def read_readme() -> str:
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "ChronicleMem"


def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    return []


setup(
    name="chronicle-mem",
    version="0.1.0",
    description="ChronicleMem: hybrid adaptive-routing retrieval for long-term personalized memory QA, built on ATM-Bench",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Kuanyan Zhu",
    author_email="zkuanyan@gmail.com",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=read_requirements(),
    include_package_data=True,
    zip_safe=False,
)
