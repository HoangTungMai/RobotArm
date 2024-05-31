# parameter_5dof.py
import numpy as np
from math import pi, cos, sin
import time

class Arm:
    def __init__(self):
        self.L0 = 28 + 50
        self.L1 = 40 # m
        self.L2 = 140
        self.L3 = 109
        self.L4 = 30
        self.L5 = 100

    def get_parameters(self):
        return self.L0, self.L1, self.L2, self.L3, self.L4, self.L5
    
    def fk(self, q1, q2, q3, q4, q5):
        L0, L1, L2, L3, L4, L5 = self.get_parameters()
    
        xE = (cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3))*L5*cos(q5) + ((-cos(q1)*cos(q2)*sin(q3) - cos(q1)*sin(q2)*cos(q3))*cos(q4) + sin(q1)*sin(q4))*L5*sin(q5) + (cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3))*L4 + cos(q1)*cos(q2)*L3*cos(q3) - cos(q1)*sin(q2)*L3*sin(q3) + cos(q1)*L2*cos(q2)   
        yE = (sin(q1)*cos(q2)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*L5*cos(q5) + ((-sin(q1)*cos(q2)*sin(q3) - sin(q1)*sin(q2)*cos(q3))*cos(q4) - cos(q1)*sin(q4))*L5*sin(q5) + (sin(q1)*cos(q2)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*L4 + sin(q1)*cos(q2)*L3*cos(q3) - sin(q1)*sin(q2)*L3*sin(q3) + sin(q1)*L2*cos(q2)
        zE = (sin(q2)*cos(q3) + cos(q2)*sin(q3))*L5*cos(q5) + (-sin(q2)*sin(q3) + cos(q2)*cos(q3))*cos(q4)*L5*sin(q5) + (sin(q2)*cos(q3) + cos(q2)*sin(q3))*L4 + sin(q2)*L3*cos(q3) + cos(q2)*L3*sin(q3) + L2*sin(q2) + L1 + L0
        
        return np.array([xE, yE, zE])
    
    def J_nd(self, q1, q2, q3, q4, q5):
        L0, L1, L2, L3, L4, L5 = self.get_parameters()
    
        J11 = (((-L5*cos(q5) - L3 - L4)*cos(q3) + cos(q4)*sin(q3)*sin(q5)*L5 - L2)*cos(q2) + sin(q2)*(cos(q3)*cos(q4)*sin(q5)*L5 + sin(q3)*(L5*cos(q5) + L3 + L4)))*sin(q1) + L5*sin(q4)*sin(q5)*cos(q1)
        J12 = -(((L5*cos(q5) + L3 + L4)*cos(q3) - cos(q4)*sin(q3)*sin(q5)*L5 + L2)*sin(q2) + (cos(q3)*cos(q4)*sin(q5)*L5 + sin(q3)*(L5*cos(q5) + L3 + L4))*cos(q2))*cos(q1)
        J13 = -cos(q1)*((cos(q3)*cos(q4)*sin(q5)*L5 + sin(q3)*(L5*cos(q5) + L3 + L4))*cos(q2) + sin(q2)*((L5*cos(q5) + L3 + L4)*cos(q3) - cos(q4)*sin(q3)*sin(q5)*L5))
        J14 = (sin(q4)*(sin(q2)*cos(q3) + cos(q2)*sin(q3))*cos(q1) + sin(q1)*cos(q4))*L5*sin(q5)
        J15 = -((cos(q4)*(sin(q2)*cos(q3) + cos(q2)*sin(q3))*cos(q5) - sin(q5)*(sin(q2)*sin(q3) - cos(q2)*cos(q3)))*cos(q1) - sin(q4)*cos(q5)*sin(q1))*L5

        J21 = (((L5*cos(q5) + L3 + L4)*cos(q3) - cos(q4)*sin(q3)*sin(q5)*L5 + L2)*cos(q2) - sin(q2)*(cos(q3)*cos(q4)*sin(q5)*L5 + sin(q3)*(L5*cos(q5) + L3 + L4)))*cos(q1) + L5*sin(q4)*sin(q5)*sin(q1)
        J22 = -(((L5*cos(q5) + L3 + L4)*cos(q3) - cos(q4)*sin(q3)*sin(q5)*L5 + L2)*sin(q2) + (cos(q3)*cos(q4)*sin(q5)*L5 + sin(q3)*(L5*cos(q5) + L3 + L4))*cos(q2))*sin(q1)
        J23 = -((cos(q3)*cos(q4)*sin(q5)*L5 + sin(q3)*(L5*cos(q5) + L3 + L4))*cos(q2) + sin(q2)*((L5*cos(q5) + L3 + L4)*cos(q3) - cos(q4)*sin(q3)*sin(q5)*L5))*sin(q1)
        J24 = -L5*sin(q5)*(-sin(q4)*(sin(q2)*cos(q3) + cos(q2)*sin(q3))*sin(q1) + cos(q1)*cos(q4))
        J25 = -((cos(q4)*(sin(q2)*cos(q3) + cos(q2)*sin(q3))*cos(q5) - sin(q5)*(sin(q2)*sin(q3) - cos(q2)*cos(q3)))*sin(q1) + sin(q4)*cos(q5)*cos(q1))*L5

        J31 = 0
        J32 = ((L5*cos(q5) + L3 + L4)*cos(q3) - cos(q4)*sin(q3)*sin(q5)*L5 + L2)*cos(q2) - sin(q2)*(cos(q3)*cos(q4)*sin(q5)*L5 + sin(q3)*(L5*cos(q5) + L3 + L4))
        J33 = ((L5*cos(q5) + L3 + L4)*cos(q3) - cos(q4)*sin(q3)*sin(q5)*L5)*cos(q2) - sin(q2)*(cos(q3)*cos(q4)*sin(q5)*L5 + sin(q3)*(L5*cos(q5) + L3 + L4))
        J34 = sin(q4)*sin(q5)*L5*(sin(q2)*sin(q3) - cos(q2)*cos(q3))
        J35 = -L5*(cos(q5)*(sin(q2)*sin(q3) - cos(q2)*cos(q3))*cos(q4) + (sin(q2)*cos(q3) + cos(q2)*sin(q3))*sin(q5))

        J = np.array([[J11, J12, J13, J14, J15],
                    [J21, J22, J23, J24, J25],
                    [J31, J32, J33, J34, J35]])
        
        Jt = J.T
        Jnd = np.dot(Jt, np.linalg.inv(np.dot(J, Jt)))
        
        return np.linalg.pinv(J)
    
    def ik(self, X_0, q_):
       
        
        ini_time = time.time()
        # Initial values of joint angles q_0
        q1_0 = q_[0]
        q2_0 = q_[1]
        q3_0 = q_[2]
        q4_0 = q_[3]
        q5_0 = q_[4]
        q6_0 = q_[5]

        # Calculate the accurate values of joint angles q_0
        for n in range(1, 10**10):
            
            Jnd_0 = self.J_nd(q1_0, q2_0, q3_0, q4_0, q5_0)  # Calculate the Jacobian inverse at q_0
            xE_0, yE_0, zE_0 = self.fk(q1_0, q2_0, q3_0, q4_0, q5_0)  # Recalculate xx_0, yy_0 based on q_0
            XX_0 = np.array([[xE_0], [yE_0], [zE_0]])
            delta_q_0 = np.dot(Jnd_0, (X_0 - XX_0))  # Calculate the correction values delta_q_0
            
            if time.time() - ini_time > 0.02:
                print("Ngoai khong gian lam viec")
                return q_
        
            # Update the joint angles q_0
            q1_0 += delta_q_0[0, 0]
            q2_0 += delta_q_0[1, 0]
            q3_0 += delta_q_0[2, 0]
            q4_0 += delta_q_0[3, 0]
            q5_0 += delta_q_0[4, 0]

            # Declare the necessary accuracy and create a loop for calculation
            ss = 10**(-4)
            if abs(delta_q_0[0, 0]) < ss and abs(delta_q_0[1, 0]) < ss and abs(delta_q_0[2, 0]) < ss and abs(delta_q_0[3, 0]) < ss and abs(delta_q_0[4, 0]) < ss:
                break

            n += 1
        q_out = np.array([q1_0, q2_0, q3_0, q4_0, q5_0, q6_0])
        
        return q_out