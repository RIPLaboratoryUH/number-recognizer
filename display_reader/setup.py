from setuptools import setup
from glob import glob
import os

package_name = 'display_reader'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='ROS2 display reader node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'display_reader_node = display_reader.display_reader_node:main',
        ],
    },
)
