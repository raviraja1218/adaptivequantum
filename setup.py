from setuptools import setup, find_packages

setup(
    name="adaptivequantum",
    version="1.0.0",
    description="ML-Driven Adaptive Error Correction and Quantum Circuit Compilation",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.10",
)
