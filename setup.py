

from setuptools import setup, find_packages

setup(
    name="codevul_plus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "networkx>=2.6.3",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "transformers>=4.18.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
    description="GCN reasoning-Enhanced Vulnerability Detection",
    author="FAST-cyber-Lab",
    author_email="jfsmrewm@gmail.com",
    url="https://github.com/FAST-cyber-Lab/CodeVulplus",
)
