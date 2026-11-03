import heapq

class Event:
    ARRIVAL = 0
    SERVICE = 1
    END_SERVICE = 2
    DEPARTURE = 3

    def __init__(self, event_type, event_time, batch, customers, server=None):
        self.event_type = event_type
        self.event_time = event_time
        self.batch = batch
        self.customers = customers
        self.server = server


    def __lt__(self, other):
        return self.event_time < other.event_time

    def __str__(self):
        return f"{self.event_type} at {self.event_time}"
