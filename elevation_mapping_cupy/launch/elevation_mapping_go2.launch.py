import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.descriptions import ParameterFile


def generate_launch_description():
    package_name = 'elevation_mapping_cupy'
    share_dir = get_package_share_directory(package_name)
    
    go2_param_path = os.path.join(share_dir, 'config', 'setups', 'go2', 'go2_param.yaml')
    go2_config_path = os.path.join(share_dir, 'config', 'setups', 'go2', 'go2.yaml')

    weight_file_path = os.path.join(share_dir, 'config', 'core', 'weights.dat')
    plugin_config_file_path = os.path.join(share_dir, 'config', 'setups', 'go2', 'go2_plugin_config.yaml')

    if not os.path.exists(go2_param_path):
        raise FileNotFoundError(f"Go2 param file not found: {go2_param_path}")
    if not os.path.exists(go2_config_path):
        raise FileNotFoundError(f"Go2 config file not found: {go2_config_path}")
    if not os.path.exists(weight_file_path):
        raise FileNotFoundError(f"Weight file not found: {weight_file_path}")
    if not os.path.exists(plugin_config_file_path):
        raise FileNotFoundError(f"Plugin config file not found: {plugin_config_file_path}")

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            share_dir, 'rviz', 'go2_tf_model_pointcloud.rviz'
        ]),
        description='Path to the RViz config file'
    )
    rviz_config = LaunchConfiguration('rviz_config')

    use_python_node_arg = DeclareLaunchArgument(
        'use_python_node',
        default_value='false',
        description='Use the Python node if true'
    )
    use_python_node = LaunchConfiguration('use_python_node')

    store_absolute_z_arg = DeclareLaunchArgument(
        'store_absolute_z',
        default_value='false',
        description='Store raw Z values coming from elevation mapping node in elevation_to_policy_node.'
    )
    store_absolute_z = LaunchConfiguration('store_absolute_z')

    elevation_mapping_node = Node(
        package='elevation_mapping_cupy',
        executable='elevation_mapping_node',
        name='elevation_mapping_node',
        output='screen',
        parameters=[
            go2_param_path,
            go2_config_path,
            {
                'use_sim_time': use_sim_time,
                'weight_file': weight_file_path,
                'plugin_config_file': plugin_config_file_path
            }
        ],
        condition=UnlessCondition(use_python_node)
    )

    elevation_mapping_node_py = Node(
        package='elevation_mapping_cupy',
        executable='elevation_mapping_node.py',
        name='elevation_mapping_node',
        output='screen',
        parameters=[
            ParameterFile(go2_param_path, allow_substs=True),
            go2_config_path,
            {'use_sim_time': use_sim_time}
        ],
        condition=IfCondition(use_python_node)
    )

    elevation_to_policy_node = Node(
        package='elevation_mapping_cupy',
        executable='elevation_to_policy_node.py',
        name='elevation_to_policy_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}, {'store_absolute_z': store_absolute_z}]
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        rviz_config_arg,
        use_python_node_arg,
        store_absolute_z_arg,
        elevation_mapping_node_py,
        elevation_mapping_node,
        elevation_to_policy_node,
        rviz_node
    ])