import numpy as np
from physics_sim import PhysicsSim

# http://www.cs.utep.edu/vladik/2013/tr13-44.pdf
def smoothabs(x):
    return np.sqrt(np.square(x) + 10e-16)

def smoothmin(a,b):
    alpha = -100
    return (a*np.exp(alpha*a) + b*np.exp(alpha*b))/(np.exp(alpha*a) + np.exp(alpha*b))
    
def mag(v):
    return np.sqrt(v.dot(v)) # supposedly faster than np.linalg.norm

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, target_vel=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        #self.state_size  = self.action_repeat * (self.sim.pose.size+self.sim.v.size+self.sim.angular_v.size)
        self.state_size  = self.action_repeat * (self.sim.pose.size + self.sim.angular_v.size)
        
        # main hold/down/up (0,1,2)
        self.action_size = 3
        
        # Set the main throttle speed so that the thrust of the four rotors combined
        # counteracts the force of gravity. We don't want much margin below that, but more margin above
        # for roll/pitch maneuvers and takeoff (a 45 degree bank requires another factor of sqrt(2)=1.414).
        # Then to first order (neglecting drag; i.e., hovering), according to the sim,
        #            individual rotor speed = sqrt(m*g/(4*C_T*rho*D^4)
        #                                   ~= 400 Hz
        
        self.speed_settings = np.array([400,300,500])

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.target_vel = target_vel if target_vel is not None else np.array([0., 0., 0.]) 
        
        self.initvec2tgt  = target_pos - init_pose[:3]
        self.suminitdist  = max((np.abs(self.initvec2tgt)).sum(),5)
        self.initdist2tgt = max(mag(self.initvec2tgt),5)
        
        self.rotor_speeds = np.ones(4)  # avoids div by 0 in sim

    def calc_reward_takeoff(self, position, velocities):
        rvec2tgt = self.target_pos - position
        dist2tgt = mag(rvec2tgt)
        
        # reward for vertical flight at takeoff
        speed = mag(velocities)
        #self.takeoff = velocities[2]/(speed*time) if (speed*time) > 0 else 0.0
        #self.takeoff = smoothmax(self.takeoff,0.)
        
        # reward/punishment for direction with respect to the target - 
        #  goes to zero close to the target
        #dircomponent = np.tanh([np.dot(rvec2tgt,velocities)/(speed)])
        #self.dircomponent = np.dot(rvec2tgt,velocities)/(speed*dist2tgt) if (speed*dist2tgt) > 0. else 0.
        self.dircomponent = (rvec2tgt[2]*velocities[2])/(speed*dist2tgt) if (speed*dist2tgt) > 0. else 0.
        
        #self.distpunishment = (smoothabs(rvec2tgt)).sum()/self.suminitdist
        self.distpunishment = abs(rvec2tgt[2])/10.
        #self.distpunishment = rvec2tgt.dot(rvec2tgt)/50.
        #distpunishment = np.tanh([(smoothabs(rvec2tgt)).sum()])

        return np.tanh([0.2*(self.dircomponent - self.distpunishment)]) # + 0.1*self.takeoff 

    def calc_reward_hover(self, position, velocities):
        rvec2tgt = self.target_pos - position
        
        self.distpunishment = smoothabs(rvec2tgt[2])
        
        return np.tanh([-self.distpunishment])
                
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        return self.calc_reward_takeoff(self.sim.pose[:3],self.sim.v)
        #return self.calc_reward_hover(self.sim.pose[:3],self.sim.v)

    def get_max_reward(self):
        return self.action_repeat*self.calc_reward_takeoff(self.target_pos,np.zeros(3))
        #return self.action_repeat*self.calc_reward_hover(self.target_pos,np.zeros(3))
    
    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        
        self.rotor_speeds = np.array([self.speed_settings[action]]*4)

        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(self.rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all += [self.sim.pose, self.sim.angular_v]
            if self.sim.crashed:
                reward = -5
                done = True
        #if (np.square(self.sim.pose[:3] - self.target_pos)).sum() < 1:  # Close enough!
            #done = True
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self,new_tgt_pos=None):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        if new_tgt_pos is not None:
            self.target_pos = new_tgt_pos 
            self.initvec2tgt = self.target_pos - self.sim.pose[:3]
            self.suminitdist = max((smoothabs(self.initvec2tgt)).sum(),5)
            self.initdist2tgt = mag(self.initvec2tgt)
        state = np.concatenate([self.sim.pose, self.sim.angular_v] * self.action_repeat) 

        return state