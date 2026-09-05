"""
setup.py — makes the qnmrd package installable via pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="qnmrd",
    version="0.1.0",
    description=(
        "Quantum-Mechanical Framework for Predicting Low-Field Paramagnetic NMRD"
    ),
    author="Daniel Conde Torres",
    packages=find_packages(exclude=["tests*", "scripts*", "archive*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "quantum": [
            "qiskit>=1.0.0",
            "qiskit-aer>=0.14.0",
            "qiskit-algorithms>=0.3.0",
        ],
        "app": [
            "plotly>=5.14.0",
            "streamlit>=1.22.0",
        ],
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)
