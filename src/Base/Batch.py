from Base.Customer import Customer


class Batch:
    def __init__(self, arrivalTime, batchSize, customers=None):
        self.arrivalTime = arrivalTime
        self.systemArrivalTime = arrivalTime
        self.batchSize = batchSize
        self.customers = customers

    def all_delivered(self):
        if not self.customers:
            return False
        return all(c.status == Customer.DELIVERED for c in self.customers)

    def all_served(self):
        if not self.customers:
            return False
        return all(c.status == Customer.PICKED or c.status == Customer.DELIVERED for c in self.customers)

    def __str__(self):
        return f"Batch arrived at time {self.arrivalTime} of size {self.batchSize}"
