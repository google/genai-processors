from setuptools import setup, find_packages

setup(
    name='genai-processors',
    version='0.0.1',  # Placeholder version, should be updated by project maintainers
    packages=find_packages(),
    install_requires=[
        'google-generativeai',
    ],
    python_requires='>=3.10',
    author='Google',  # Inferred from the original repo
    description='Build Modular, Asynchronous, and Composable AI Pipelines for Generative AI.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Mirza-Samad-Ahmed-Baig/genai-processors', # Your forked repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)
