from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]


setup(
    name="swarmmind",
    version="1.1.0",
    description="安全、可控、智能的多 Agent 协作框架",
    author="SwarmMind Team",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        "console_scripts": [
            "swarmmind=swarmmind.cli.main:run",
        ],
    },
    python_requires=">=3.10",
)
