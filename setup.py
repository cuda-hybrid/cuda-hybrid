import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cuda-hybrid-CUDA-HYBRID",
    version="0.1.5",
    description="Run ABM/FCM models on CUDA cores",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cuda-hybrid/cuda-hybrid",
    project_urls={
        "Bug Tracker": "https://github.com/cuda-hybrid/cuda-hybrid/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=["setuptools>=42",
                    "networkx >= 2.6.3",
                    "nptyping >= 1.4.4",
                    "numba >= 0.48.0",
                    "numpy >= 1.21.5"],
)