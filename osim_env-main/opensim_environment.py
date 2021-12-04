'''Adapted version of osim-rl repository. There are some things that
we would like to implement differently.

Author: Dimitar Stanev (dimitar.stanev@epfl.ch)

Changelog:

01/01/2020: change coordinate set in state description

28/02/2020: remove redundant functions, make use of Actuators so that
we can use torque actuated models, implement save simulation

03/07/2020: remove custom environment and include initial velocity
setter, rename also init_pose to set_coordinates, remove redundant
things from the environment

09/07/2020: augment the state dictionary, default control = 0 (fix
bug)
'''
# %%
import opensim
#import gym
#from gym import spaces
import numpy as np
from flatten_dict import flatten


class Specification:
    timestep_limit = None

    def __init__(self, timestep_limit):
        self.timestep_limit = timestep_limit


class Spec(object):
    def __init__(self, *args, **kwargs):
        self.id = 0
        self.timestep_limit = 300


class OsimModel(object):
    def __init__(self, model_path, step_size, integrator_accuracy, body_ext_force, visualize, save_kin):
        self.integrator_accuracy = integrator_accuracy
        self.model = opensim.Model(model_path)
        self.save_kin = save_kin
        if save_kin is not None:
            self.body_kinematics = opensim.BodyKinematics()
            self.body_kinematics.setModel(self.model)
            self.model.addAnalysis(self.body_kinematics)
            self.reported_forces = opensim.ForceReporter()
            self.reported_forces.setModel(self.model)
            self.model.addAnalysis(self.reported_forces)

        self.model_state = self.model.initSystem()
        self.model.setUseVisualizer(visualize)
        self.brain = opensim.PrescribedController()
        self.istep = 0
        self.step_size = step_size
        self.step_alive = 0

        # get model sets
        self.actuator_set = self.model.getActuators()
        self.force_set = self.model.getForceSet()
        self.body_set = self.model.getBodySet()
        self.coordinate_set = self.model.getCoordinateSet()
        self.marker_set = self.model.getMarkerSet()

        # define controller
        self.action_min = []
        self.action_max = []
        for j in range(self.actuator_set.getSize()):
            func = opensim.Constant(0.0)
            actuator = self.actuator_set.get(j)
            self.brain.addActuator(actuator)
            self.brain.prescribeControlForActuator(j, func)
            scalar_actuator = opensim.ScalarActuator_safeDownCast(actuator)
            if scalar_actuator is None:
                # currently only BodyActuator is not handled (not crucial)
                # https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Actuator.html
                raise RuntimeError('un-handled type of scalar actuator')
            else:
                self.action_min.append(scalar_actuator.getMinControl())
                self.action_max.append(scalar_actuator.getMaxControl())

        # add potential external force
        if body_ext_force is not None:
            prescribed_force = opensim.PrescribedForce("perturbation", self.body_set.get(body_ext_force))
            prescribed_force.setPointFunctions(opensim.Constant(0), opensim.Constant(0), opensim.Constant(0))
            prescribed_force.setForceFunctions(opensim.Constant(0), opensim.Constant(0), opensim.Constant(0))
            self.model.addForce(prescribed_force)

        self.action_space_size = self.actuator_set.getSize()
        self.model.addController(self.brain)
        self.model_state = self.model.initSystem()


    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError('NaN passed in the activation vector. ')

        # might have torque actuators
        action = np.clip(np.array(action), self.action_min, self.action_max)
        self.last_action = action

        brain = opensim.PrescribedController.safeDownCast(
            self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue(float(action[j]))

    def get_last_action(self):
        return self.last_action

    def get_state_dict(self):
        self.model.realizeAcceleration(self.state)
        obs = {}

        # coordinates
        obs['coordinate_pos'] = {}
        obs['coordinate_vel'] = {}
        obs['coordinate_acc'] = {}
        for i in range(self.coordinate_set.getSize()):
            coordinate = self.coordinate_set.get(i)
            name = coordinate.getName()
            obs['coordinate_pos'][name] = coordinate.getValue(self.state)
            obs['coordinate_vel'][name] = coordinate.getSpeedValue(self.state)
            obs['coordinate_acc'][name] = coordinate.getAccelerationValue(self.state)

        # bodies
        obs['body_pos'] = {}
        obs['body_vel'] = {}
        obs['body_acc'] = {}
        obs['body_pos_rot'] = {}
        obs['body_vel_rot'] = {}
        obs['body_acc_rot'] = {}
        for i in range(self.body_set.getSize()):
            body = self.body_set.get(i)
            name = body.getName()
            obs['body_pos'][name] = [body.getTransformInGround(self.state).p()[i] for i in range(3)] #getPositionInGround(self.state)[i]
            obs['body_vel'][name] = [body.getVelocityInGround(
                self.state).get(1).get(i) for i in range(3)]
            obs['body_acc'][name] = [body.getAccelerationInGround(
                self.state).get(1).get(i) for i in range(3)]

            obs['body_pos_rot'][name] = [body.getTransformInGround(
                self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            obs['body_vel_rot'][name] = [body.getVelocityInGround(
                self.state).get(0).get(i) for i in range(3)]
            obs['body_acc_rot'][name] = [body.getAccelerationInGround(
                self.state).get(0).get(i) for i in range(3)]

        # mass center
        obs['body_pos']['center_of_mass'] = [
            self.model.calcMassCenterPosition(self.state)[i] for i in range(3)]
        obs['body_vel']['center_of_mass'] = [
            self.model.calcMassCenterVelocity(self.state)[i] for i in range(3)]
        obs['body_acc']['center_of_mass'] = [
            self.model.calcMassCenterAcceleration(self.state)[i] for i in range(3)]

        # forces
        obs['forces'] = {}
        obs['contact_forces'] = {}
        obs['coordinate_limit_forces'] = {}
        obs['scalar_actuator_forces'] = {}
        for i in range(self.force_set.getSize()):
            force = self.force_set.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            # we check the type of force for quick access
            contact_force = opensim.HuntCrossleyForce_safeDownCast(force)
            coordinate_limit_force = opensim.CoordinateLimitForce_safeDownCast(force)
            scalar_actuator = opensim.ScalarActuator_safeDownCast(force)
            if contact_force:
                # It is assumed that the first 6 values is the total
                # wrench (force, moment) applied on the ground plane
                # (they must be negated). The rest are the forces
                # applied on individual points.
                 obs['contact_forces'][name] = [-values.get(0), -values.get(1), -values.get(2),
                                                -values.get(3), -values.get(4), -values.get(5)]
            elif coordinate_limit_force:
                # coordinate limiting forces return two values, but
                # only one is active (non-zero) or both are zero
                if values.get(0) == 0:
                    value = values.get(1)
                else:
                    value = values.get(0)

                obs['coordinate_limit_forces'][name] = value
            elif scalar_actuator:
                obs['scalar_actuator_forces'][name] = values.get(0)
            else:
                obs['forces'][name] = [values.get(i) for i in range(values.size())]

        # muscles (model might be torque actuated)
        if self.model.getMuscles().getSize() != 0:
            obs['muscles'] = {}
            for i in range(self.model.getMuscles().getSize()):
                muscle = self.model.getMuscles().get(i)
                name = muscle.getName()
                obs['muscles'][name] = {}
                obs['muscles'][name]['activation'] = muscle.getActivation(
                    self.state)
                obs['muscles'][name]['fiber_length'] = muscle.getFiberLength(
                    self.state)
                obs['muscles'][name]['fiber_velocity'] = muscle.getFiberVelocity(
                    self.state)
                obs['muscles'][name]['fiber_force'] = muscle.getFiberForce(
                    self.state)
                obs['muscles'][name]['force_length'] = muscle.getActiveForceLengthMultiplier(
                    self.state)
                obs['muscles'][name]['force_velocity'] = muscle.getForceVelocityMultiplier(
                    self.state)
                obs['muscles'][name]['passive_force'] = muscle.getPassiveForceMultiplier(
                    self.state)
                obs['muscles'][name]['cos_pennation_angle'] = muscle.getCosPennationAngle(
                    self.state)
                obs['muscles'][name]['fmax'] = muscle.getMaxIsometricForce()

        # markers
        obs['markers'] = {}
        for i in range(self.marker_set.getSize()):
            marker = self.marker_set.get(i)
            name = marker.getName()
            obs['markers'][name] = {}
            obs['markers'][name]['pos'] = [
                marker.getLocationInGround(self.state)[i] for i in range(3)]
            obs['markers'][name]['vel'] = [
                marker.getVelocityInGround(self.state)[i] for i in range(3)]
            obs['markers'][name]['acc'] = [
                marker.getAccelerationInGround(self.state)[i] for i in range(3)]

        # muscle moment arm for each coordinate
        if self.moments is not None:
            obs['coordinate_muscle_moment_arm'] = {}
            for i in range(len(self.moments)):
                coordinate = self.coordinate_set.get(self.moments[i])
                name = coordinate.getName()
                obs['coordinate_muscle_moment_arm'][name] = 0  # {}
                if self.model.getMuscles().getSize() != 0:
                    for i in range(self.model.getMuscles().getSize()):
                        muscle = self.model.getMuscles().get(i)
                        name_muscle = muscle.getName()
                        force = self.model.getForceSet().get(name_muscle)
                        #muscle_ = opensim.Thelen2003Muscle.safeDownCast(force)
                        muscle_ = opensim.Millard2012EquilibriumMuscle.safeDownCast(force)
                        coord = self.model.getCoordinateSet().get(name)
                        force = obs['muscles'][name_muscle]['fiber_force']
                        obs['coordinate_muscle_moment_arm'][name] += force*muscle_.computeMomentArm(self.state, coord)  # [name_muscle]

        return obs

    def get_action_space_size(self):
        return self.action_space_size

    def reset_manager(self):
        self.manager = opensim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def reset(self):
        self.state = self.model.initializeState()
        self.model.equilibrateMuscles(self.state)
        self.state.setTime(0)
        self.istep = 0
        self.step_alive = 0
        self.reset_manager()
        if self.save_kin is not None:
            self.body_kinematics.getPositionStorage().reset(0)
            self.body_kinematics.getVelocityStorage().reset(0)
            self.body_kinematics.getAccelerationStorage().reset(0)
            self.reported_forces.getForceStorage().reset(0)

    def integrate(self):
        self.istep += 1
        self.state = self.manager.integrate(self.step_size * self.istep)
        self.step_alive += 1

    def set_time(self, t):
        self.state.setTime(t)
        self.istep = int(self.state.getTime() / self.step_size)
        self.reset_manager()

    def set_coordinates(self, q_dict):
        '''Set coordinate values.

        Parameters
        ----------

        q_dict: a dictionary containing the coordinate names and
        values in rad or m.

        '''
        for coordinate, value in q_dict.items():
            self.coordinate_set.get(coordinate).setValue(self.state, value)

        self.reset_manager()

    def set_velocities(self, u_dict):
        '''Set coordinate velocities.

        Parameters
        ----------

        u_dict: a dictionary containing the coordinate names and
        velocities in rad/s or m/s.

        '''
        for coordinate, value in u_dict.items():
            self.coordinate_set.get(coordinate).setSpeedValue(self.state, value)

        self.reset_manager()

    def save_simulation(self, base_dir):
        '''Saves simulation files into base_dir.'''
        self.manager.getStateStorage().printToFile(base_dir + '/simulation_States.sto', 'w', '')
        self.body_kinematics.printResults('body_simulation', base_dir)
        self.reported_forces.printResults('reported_force', base_dir)
