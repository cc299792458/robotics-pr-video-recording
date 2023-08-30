from .base_env import BaseEnv
import sapien

class DrillScrew(BaseEnv):
    def __init__(self):
        super().__init__()
    
    def _add_workspace(self):
        pass

    def _add_actor(self):
        # TODO(haoyang): add task related actors here.
        """
        -"kitchen_env",
        -"kitchen_table",
        -"dishes",
        -""
        """
        # add the kitchen_env
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(filename='./assets/Kitchen/kitchen1/Kitchen_tietu.dae')
        builder.add_nonconvex_collision_from_file(filename='./prep/CoACD/outputs/Kitchen1.obj')
        kitchen_env = builder.build_kinematic(name='kitchen_env') # can not be affected by external forces
        kitchen_env.set_pose(sapien.Pose(p=[-6.7, -3.4, -1.0+0.23],q=[0.707, 0, 0, -0.707]))

        # add the kitchen_table
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(filename='./assets/Kitchen/Kitchen_table/table_sink.dae')
        builder.add_nonconvex_collision_from_file(filename='./prep/CoACD/outputs/table_sink.obj')
        kitchen_table = builder.build_kinematic(name='kitchen_table') # can not be affected by external forces
        kitchen_table.set_pose(sapien.Pose(p=[0.30, 1.0, -1.0],q=[0.707, 0, 0, -0.707]))

        # add the dishes
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(filename='./assets/Kitchen/plates/plate.dae')
        builder.add_collision_from_file(filename='./prep/CoACD/outputs/plate.obj')
        for i in range(5):
            name = "dishes_" + str(i)
            dishes = builder.build(name = name) # can not be affected by external forces
            dishes.set_pose(sapien.Pose(p=[-1.179, 0.8+0.2*i, -0.231+0.5],q=[-0.291, 0, 0, 0.957]))

        # add the photo
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(filename='./assets/Kitchen/photo/photo.dae')
        builder.add_collision_from_file(filename='./assets/Kitchen/photo/photo.obj')
        photo = builder.build(name = 'photo') # can not be affected by external forces
        photo.set_pose(sapien.Pose(p=[-1.139, -1.686, -0.223],q=[-0.018, 0.024, 0.698, -0.715]))

        # add the wall
        builder = self._scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.05, 3.0, 3.5])
        wall = builder.build_kinematic(name = 'wall') # can not be affected by external forces
        wall.set_pose(sapien.Pose(p=[0.350, 2.294, -1],q=[0.707, 0, 0, -0.707]))

        # add the drill

        # add the sponge

        # add the dish rack