from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()

requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="future_stars",
    version="1.0.0",
    description="AI model to predict Future Star players based on performance data",
    license="MIT",
    url="https://github.com/rahafw/future_stars",
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
)
