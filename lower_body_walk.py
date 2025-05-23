import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from assets import ASSET_DIR
import os
import crocoddyl
from crocoddyl.utils.biped import SimpleBipedGaitProblem


class LowerBodyWalk(SimpleBipedGaitProblem):
    def __init__(self, rmodel, right_foot, left_foot,integrator="euler", control="zero", fwddyn=True):
        super().__init__(rmodel, right_foot, left_foot,integrator, control, fwddyn)

        self.get_body_joints()
        self.upper_body_reference_state = self._create_upper_body_reference_state()


    def get_body_joints(self):
        lower_body = {}
        # NOTE: Now waist is not included in lower body
        lower_body_keys = ['ankle', 'knee', 'hip']
        upper_body = {}
        # get lower and upper body joints
        for idx, joint in enumerate(self.rmodel.joints):
            joint_name = self.rmodel.names[idx]
            if any(key in joint_name for key in lower_body_keys):
                lower_body[idx] = joint_name
            else:
                upper_body[idx] = joint_name

        self.lower_body = lower_body
        self.upper_body = upper_body
        return lower_body, upper_body
    


    def _create_upper_body_reference_state(self):
        """Create reference state with upper body at default configuration"""
        nq = self.rmodel.nq
        nv = self.rmodel.nv
        ref_state = np.zeros(nq + nv)

        # Set upper body positions from half_sitting
        q_ref = self.rmodel.referenceConfigurations["half_sitting"]
        ref_state[:nq] = q_ref  

        # Set upper body velocities to zero
        for jid in self.upper_body:
            jmodel = self.rmodel.joints[jid]
            if jmodel.nv == 0:
                continue
            v_start = jmodel.idx_v
            for i in range(jmodel.nv):
                ref_state[nq + v_start + i] = 0.0
        return ref_state


    def get_joint_weights(self, upper_body_weight=1e2, residual_nr=None):
        """Get state weights with selective upper/lower body weighting
        
        Structure: [0]*3 + [500.0]*3 + [joint_positions] + [joint_velocities]
        Where joint_positions and joint_velocities are weighted by body part
        """
        if residual_nr is None:
            temp_ref_state = np.zeros(self.state.nx)
            temp_residual = crocoddyl.ResidualModelState(self.state, temp_ref_state, 0)
            nr = temp_residual.nr
        else:
            nr = residual_nr
        
        nq = self.state.nq
        nv = self.state.nv
        
        stateWeights = np.zeros(nr)
        
        # floating base
        if nr >= 6:
            stateWeights[0:3] = 0.0      # position x,y,z
            stateWeights[3:6] = 500.0    # orientation roll,pitch,yaw
        
        # Joint positions part: index 6 to (nq-1) in residual space
        joint_pos_start = 6
        joint_pos_end = min(nq - 1, nr)  # -1 for quaternion compression
        
        if joint_pos_end > joint_pos_start:
            # lower body weight
            stateWeights[joint_pos_start:joint_pos_end] = 0.01  # Default lower body weight
            
            # upper body weight
            for jid in self.upper_body:
                jmodel = self.rmodel.joints[jid]
                if jmodel.nq == 0:
                    continue

                q_start = jmodel.idx_q
                for i in range(jmodel.nq):
                    idx = q_start + i
                    if idx > 6:  
                        residual_idx = idx - 1 
                        if joint_pos_start <= residual_idx < joint_pos_end:
                            stateWeights[residual_idx] = upper_body_weight
        
        # Joint velocities part: starts after positions
        vel_start = nq - 1  
        vel_end = min(vel_start + nv, nr)
        
        if vel_end > vel_start:
            # lower body weight
            stateWeights[vel_start:vel_end] = 10.0  # Default velocity weight
            
            # upper body weight
            for jid in self.upper_body:
                jmodel = self.rmodel.joints[jid]
                if jmodel.nv == 0:
                    continue
                
                v_start = jmodel.idx_v
                for i in range(jmodel.nv):
                    residual_v_idx = vel_start + v_start + i
                    if vel_start <= residual_v_idx < vel_end:
                        stateWeights[residual_v_idx] = upper_body_weight
        
        return stateWeights


    
    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None):
        """Create swing foot model with upper body constraints"""
        # Creating a 6D multi-contact model, and then including the supporting foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)
            
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                i,
                pin.SE3.Identity(),
                pin.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 30.0]),
            )
            contactModel.addContact(
                self.rmodel.frames[i].name + "_contact", supportContactModel
            )
        
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        
        # CoM tracking cost
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, 1e6)
        
        # Wrench cone constraints for support feet
        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(
                self.state, wrenchActivation, wrenchResidual
            )
            costModel.addCost(
                self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1
            )
        
        # Swing foot tracking cost
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], nu
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, framePlacementResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6
                )
        
        # state residual
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.upper_body_reference_state, nu
        )
        
        # state weights
        stateWeights = self.get_joint_weights(residual_nr=stateResidual.nr)
        
        # state activation
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1e1)
        
        # Control regularization with custom weights
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        
        
        # Creating the action model for the KKT dynamics
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )
        
        # Control parametrization
        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.four
            )
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.three
            )
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)
        
        # Integration scheme
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.four, timeStep
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.three, timeStep
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.two, timeStep
            )
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        
        return model
    


    def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
        """Create pseudo-impulse model with upper body constraints"""
        # Similar structure to createSwingFootModel but for pseudo-impulse
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)
            
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                i,
                pin.SE3.Identity(),
                pin.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(
                self.rmodel.frames[i].name + "_contact", supportContactModel
            )
        
        # Creating the cost model
        costModel = crocoddyl.CostModelSum(self.state, nu)
        
        # Wrench cone constraints
        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(
                self.state, wrenchActivation, wrenchResidual
            )
            costModel.addCost(
                self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1
            )
        
        # Swing foot tasks
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], nu
                )
                frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    i[0],
                    pin.Motion.Zero(),
                    pin.LOCAL_WORLD_ALIGNED,
                    nu,
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, framePlacementResidual
                )
                impulseFootVelCost = crocoddyl.CostModelResidual(
                    self.state, frameVelocityResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e8
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_impulseVel",
                    impulseFootVelCost,
                    1e6,
                )
        
        # state residual
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.upper_body_reference_state, nu
        )
        
        # state weights
        stateWeights = self.get_joint_weights(residual_nr=stateResidual.nr)
        
        # state activation
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1e1)
        
        # Control regularization
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)
        
        
        # Creating the differential action model
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )
        
        # Integration (with zero time step for pseudo-impulse)
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.four, 0.0
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.three, 0.0
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.two, 0.0)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        
        return model
    

    def createImpulseModel(self, supportFootIds, swingFootTask, JMinvJt_damping=1e-12, r_coeff=0.0):
        """Action model for impulse models with upper body constraints.
        
        An impulse model consists of describing the impulse dynamics against a set of
        contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param JMinvJt_damping: damping parameter for the impulse model
        :param r_coeff: restitution coefficient
        :return impulse action model
        """
        # Creating a 6D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel6D(
                self.state, i, pin.LOCAL_WORLD_ALIGNED
            )
            impulseModel.addImpulse(
                self.rmodel.frames[i].name + "_impulse", supportContactModel
            )
        
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        
        # Swing foot tracking cost
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], 0
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, framePlacementResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e8
                )
        # state residual
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.upper_body_reference_state, 0
        )
        
        # state weights
        stateWeights = self.get_joint_weights(residual_nr=stateResidual.nr)
        
        # state activation
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1e1)
        
        # Creating the action model for the impulse dynamics
        model = crocoddyl.ActionModelImpulseFwdDynamics(
            self.state, impulseModel, costModel
        )
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model