from .base_env import BaseEnv
import sapien.core as sapien


class TidyUpDish(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_workspace(self):
        # add kitchen_envs
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(filename='./assets/kitchen/kitchen1/kitchen_tietu.dae')
        builder.add_multiple_collisions_from_file(scale=[1.0,1.0,0.965], filename='./assets/outputs/kitchen1.obj')
        kitchen_env = builder.build_kinematic(name='kitchen_env') # can not be affected by external forces
        kitchen_env.set_pose(sapien.Pose(p=[-6.7, -3.4225, -1.0+0.23],q=[0.707, 0, 0, -0.707]))

        # add wall_right
        builder = self._scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.05, 3.0, 3.5])

        # add wall_left
        builder = self._scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.05, 3.0, 3.5])
        wall_left = builder.build_kinematic(name = 'wall_left') # can not be affected by external forces
        wall_left.set_pose(sapien.Pose(p=[0.350, -3.580, -1],q=[0.707, 0, 0, -0.707]))

        # add sink
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(filename='./assets/kitchen/sink/stainless_steel_sink.dae')
        builder.add_multiple_collisions_from_file(filename='./assets/kitchen/sink/stainless_steel_sink.obj')
        sink = builder.build(name='sink') # can not be affected by external forces
        sink.set_pose(sapien.Pose(p=[-1.058, 0.0, -0.0096],q=[0.707, 0, 0, -0.707]))

        # add left_table 
        self._add_table(
            pose=sapien.Pose(p=[0.35, -2.2, -0.10],q=[0.707, 0, 0, -0.707]),
            length=0.6,
            width=1.8, 
            height=0.87, 
            thickness=0.03, 
            color=(0.8, 0.6, 0.4), 
            name='left_table')

    def _add_actor(self):
        """
        -"kitchen_env",
        -"kitchen_table",
        -"dishes",
        -""
        """
        # add kitchen_table
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(filename='./assets/kitchen/kitchen_table/table_sink.dae')
        builder.add_multiple_collisions_from_file(filename='./assets/outputs/table_sink.obj')
        kitchen_table = builder.build_kinematic(name='kitchen_table') # can not be affected by external forces
        kitchen_table.set_pose(sapien.Pose(p=[0.30, 0.9, -1.0],q=[0.707, 0, 0, -0.707]))

        # add dishes
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(filename='./assets/kitchen/plates/plate.dae')
        builder.add_collision_from_file(filename='./assets/outputs/plate.obj')
        for i in range(2):
            name = "dishes_" + str(i)
            dishes = builder.build(name = name) # can not be affected by external forces
            dishes.set_pose(sapien.Pose(p=[-1.057, 0.03075, 0.006+0.1*i],q=[-0.291, 0, 0, 0.957]))

        # add photo
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(filename='./assets/kitchen/photo/photo.dae')
        builder.add_collision_from_file(filename='./assets/kitchen/photo/photo.obj')
        photo = builder.build(name = 'photo') # can not be affected by external forces
        photo.set_pose(sapien.Pose(p=[-1.139, -2.108, -0.223],q=[-0.018, 0.024, 0.698, -0.715]))

        # add drill
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(scale=[0.1904,0.1904,0.1904],filename='./assets/kitchen/Drill/drill.dae')
        builder.add_multiple_collisions_from_file(scale=[0.1904,0.1904,0.1904],filename='./assets/kitchen/Drill/drill.obj')
        drill = builder.build(name = 'drill') # can not be affected by external forces
        drill.set_pose(sapien.Pose(p=[0.019, -2.395, 0.064],q=[0.707, 0, 0, -0.707]))

        # add sponge
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(scale=[0.15,0.15,0.15],filename='./assets/kitchen/sponge/sponge.dae')
        builder.add_collision_from_file(scale=[0.15,0.15,0.15],filename='./assets/kitchen/sponge/sponge.obj')
        sponge = builder.build(name = 'sponge') # can not be affected by external forces
        sponge.set_pose(sapien.Pose(p=[-1.02, 0.509, 0.0],q=[0, -0.707, 0.707, 0]))

        # add dish rack
        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(filename='./assets/kitchen/Dishrack/dishrack.dae')
        builder.add_multiple_collisions_from_file(filename='./assets/kitchen/Dishrack/dishrack.obj')
        dish_rack = builder.build(name = 'dish_rack') # can not be affected by external forces
        dish_rack.set_pose(sapien.Pose(p=[-0.975, -0.601, 0.020],q=[0, 0, 0.707, 0.707]))