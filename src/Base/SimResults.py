from collections import deque
from numpy import zeros

class SimResults:

    def __init__(self):
        self.oldTime = 0
        self.sumW = 0
        self.sumW2 = 0
        self.nW = 0
        self.waitingTimes = deque()

    def registerWaitingTime(self, w):
        self.waitingTimes.append(w)
        self.nW += 1
        self.sumW += w
        self.sumW2 += w * w

    def getMeanWaitingTime(self):
        return self.sumW / self.nW

    def getWaitingTimes(self):
        return self.waitingTimes

    def getVarianceWaitingTime(self):
        m2 = self.sumW2 / self.nW
        m1 = self.getMeanWaitingTime()
        return m2 - m1 * m1

    def get_all_times(self):
        return list(self.waitingTimes)

    def __str__(self):
        s = 'Mean time: ' + str(self.getMeanWaitingTime()) + '\n'
        s += 'Variance time: ' + str(self.getVarianceWaitingTime()) + '\n'
        return s


