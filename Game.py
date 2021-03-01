import numpy as np
class Game:
    def __init__(self):
        self.agentA_points = np.array([[0.0,1.0,10.0],[1.0,1.0,7.0],[1.0,0.0,0.0]])
        self.agentB_points = np.array([[1.0,0.0,10.0],[1.0,1.0,7.0],[0.0,1.0,0.0]])
        self.coop_bound = 0.5
        self.defect_bound = 0.5

    def determine_payoffPlanes(self):
        self.agentA_payoff = self.find_payoffPlane(self.agentA_points)
        self.agentB_payoff = self.find_payoffPlane(self.agentB_points)

    def find_payoffPlane(self, points):
        v1 = points[0,:] - points[1,:]
        v2 = points[2,:] - points[1,:]

        a,b,c = np.cross(v1,v2)

        d = np.dot(np.array([a,b,c]),points[0,:])

        return np.array([a,b,c,d])

    def find_payoff(self, xA, xB):
        payoff_A = (self.agentA_payoff[3] - self.agentA_payoff[0]*xA - self.agentA_payoff[1]*xB)/self.agentA_payoff[2]
        payoff_B = (self.agentB_payoff[3] - self.agentB_payoff[0]*xA - self.agentB_payoff[1]*xB)/self.agentB_payoff[2]

        return payoff_A,payoff_B

    def determine_solutionType(self, x1, x2):
        solution = 2
        if x1 >= self.coop_bound and x2 >= self.coop_bound:
            solution = 1

        if x1 < self.defect_bound  and x2 <= self.defect_bound:
            solution = 0

        return solution