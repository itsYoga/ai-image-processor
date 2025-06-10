from setuptools import setup, find_packages

setup(
    name="image_processing_app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-image",
        "scipy",
        "torch",
        "gradio"
    ],
) 