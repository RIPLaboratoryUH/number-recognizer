cmake_minimum_required(VERSION 3.8)
project(display_reader)

find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DisplayReading.msg"
  DEPENDENCIES std_msgs
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install the node script
install(PROGRAMS
  scripts/display_reader_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install model files
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

ament_package()
