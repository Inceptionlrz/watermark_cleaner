from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="watermark-cleaner",
    version="0.1.0",
    description="AI 暗水印检测与清洗工具",
    author="Watermark Cleaner Team",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "watermark-cleaner=watermark_cleaner.cli:main",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)