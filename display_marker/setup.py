from setuptools import setup
from glob import glob
import os

package_name = 'display_marker'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Visualize display readings as markers at robot positions',
    license='MIT',
    entry_points={
        'console_scripts': [
            'display_marker_node = display_marker.display_marker_node:main',
        ],
    },
)
