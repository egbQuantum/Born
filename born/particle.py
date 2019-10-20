# Particle swarm optimizer

import numpy as np
import random
from math import pi


class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-0.25,0.25))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, n, target, costFunc, shots):
        self.err_i=costFunc(n, target, self.position_i, shots)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.75       # constant inertia weight (how much to weigh the previous velocity)
        c1=2        # cognative constant
        c2=2        # social constant
        
        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()
            
            # We need to account for the fact that parameters are angles
            dist_cog = self.pos_best_i[i]-self.position_i[i]
            if dist_cog > pi:
                dist_cog -= 2*pi
            if dist_cog < -pi:
                dist_cog += 2*pi
                
            dist_soc = pos_best_g[i] - self.position_i[i]
            if dist_soc > pi:
                dist_soc -= 2*pi
            if dist_soc < -pi:
                dist_soc += 2*pi

            vel_cognitive=c1*r1*dist_cog
            vel_social=c2*r2*dist_soc
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            #account for particles crossing the 0=2pi line
            if self.position_i[i] > 2*pi:
                self.position_i[i] -= 2*pi
            
            if self.position_i[i] < 0:
                self.position_i[i] += 2*pi
    
                
class PSO():
    def __init__(self,n,shots,target,costFunc,x0,num_particles,maxiter):
        global num_dimensions

        num_dimensions=len(x0)
        self.err_best_g=-1                   # best error for group
        self.pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(n, target, costFunc, shots)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g=list(swarm[j].position_i)
                    self.err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(self.pos_best_g)
                swarm[j].update_position()
            i+=1
            
            if i%2 == 0:
                print("Iteration %d: %f" % (i, self.err_best_g))

        # print final results
        print('FINAL: %f' % self.err_best_g)
        
    
    def best(self):
        return self.pos_best_g
        

if __name__ == "__PSO__":
    main()