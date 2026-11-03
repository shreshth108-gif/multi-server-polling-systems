class Customer:
    WAITING = 0
    BEING_SERVED = 1
    PICKED = 2
    DELIVERED = 3

    def __init__(self, arrivalTime, location, status, batch):
        self.arrivalTime = arrivalTime
        self.systemArrivalTime = arrivalTime
        self.location = location
        self.status = status
        self.batch = batch

    def __str__(self):
        return f"Customer arrived at time {self.arrivalTime} in location {self.location}, status: {self.status}"