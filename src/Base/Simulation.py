import numpy as np
import heapq
from Base.FES import FES
from Base.Event import Event
from Base.Server import Server
from Base.Customer import Customer
from Base.Batch import Batch
from Base.SimResults import SimResults
from Base.Clustering import Clustering
from collections import deque


class Simulation:

    def __init__(self, interArrivalDist, batchDist, locationDist, serviceDist, numServers,
                 serverSpeed, serverLocation, serverRotation):
        self.interArrivalDist = interArrivalDist
        self.batchDist = batchDist
        self.locationDist = locationDist
        self.serviceDist = serviceDist
        self.numServers = numServers
        self.serverSpeed = serverSpeed
        self.serverLocation = serverLocation
        self.serverRotation = serverRotation

    @staticmethod
    def try_assign_service(servers, customersQueue, fes, current_time):
        idle_servers = [s for s in servers if s.status == Server.IDLE]
        waiting_customers = [c for c in customersQueue if c.status == Customer.WAITING]

        if not idle_servers or not waiting_customers:
            return  # nothing to do

        potential_assignments = []
        for server in idle_servers:
            for customer in waiting_customers:
                if customer.arrivalTime <= current_time + server.get_time_to_reach_location(customer.location):
                    eta = server.get_time_to_reach_location(customer.location)
                    potential_assignments.append((eta, server, customer))

        potential_assignments.sort(key=lambda x: x[0])

        final_assignments = []
        assigned_servers = set()
        assigned_customers = set()

        for eta, server, customer in potential_assignments:
            if server not in assigned_servers and customer not in assigned_customers:
                final_assignments.append((server, customer))
                assigned_servers.add(server)
                assigned_customers.add(customer)

        for server, customer in final_assignments:
            service_start = current_time + server.get_time_to_reach_location(customer.location)
            if server.items_in_hand and server.crossed_depot(customer.location):
                dep_start = current_time + server.get_time_to_reach_location(0.0)
                server.status = Server.AT_DEPOT
                fes.add(Event(Event.DEPARTURE, dep_start, customer.batch, server.items_in_hand, server))
            elif customer.arrivalTime <= service_start:
                customer.status = Customer.BEING_SERVED
                server.status = Server.TRAVELING
                fes.add(Event(Event.SERVICE, service_start, customer.batch, customer, server))

    @staticmethod
    def schedule_next_departure(servers, fes, current_time):
        for server in servers:
            if server.status == Server.IDLE and server.items_in_hand:
                time_next_event = float('inf')
                if not fes.isEmpty():
                    time_next_event = fes.events[0].event_time
                time_to_depot = server.get_time_to_reach_location(0.0)
                dep_start = current_time + time_to_depot
                if dep_start < time_next_event:
                    server.status = Server.AT_DEPOT
                    fes.add(Event(Event.DEPARTURE, dep_start, None, server.items_in_hand, server))

    @staticmethod
    def schedule_next_event(servers, customers_queue, fes, current_time):
        # 1. Find the next potential service event
        idle_servers = [s for s in servers if s.status == Server.IDLE]
        waiting_customers = [c for c in customers_queue if c.status == Customer.WAITING]

        next_service = None
        min_eta = float('inf')

        if idle_servers and waiting_customers:
            for server in idle_servers:
                for customer in waiting_customers:
                    eta = server.get_time_to_reach_location(customer.location)
                    if eta < min_eta and customer.arrivalTime <= current_time + eta:
                        min_eta = eta
                        # This is the best potential encounter so far
                        next_service = {'time': current_time + eta, 'server': server, 'customer': customer}

        # 2. Find the next potential "departure" event
        next_departure = None
        min_dep_time = float('inf')
        for server in servers:
            if server.status == Server.IDLE and server.items_in_hand:
                dep_time = current_time + server.get_time_to_reach_location(0.0)
                if dep_time < min_dep_time:
                    min_dep_time = dep_time
                    next_departure = {'time': dep_time, 'server': server}

        # 3. Determine the next event and schedule it
        # Find the time of the next event already scheduled in the FES
        next_fes_event_time = float('inf')
        if not fes.isEmpty():
            next_fes_event_time = fes.events[0].event_time

        # If the best service is the next thing to happen in the whole system
        if next_service and next_service['time'] < next_fes_event_time and next_service['time'] < min_dep_time:
            s, c = next_service['server'], next_service['customer']
            s.status = Server.TRAVELING
            fes.add(Event(Event.SERVICE, next_service['time'], c.batch, c, s))

        # If the best departure is the next thing to happen
        elif next_departure and next_departure['time'] < next_fes_event_time:
            s = next_departure['server']
            s.status = Server.AT_DEPOT
            fes.add(Event(Event.DEPARTURE, next_departure['time'], None, s.items_in_hand, s))


    def simulate(self, threshold_deliveries):

        num_delivered = 0
        fes = FES()
        t = 0

        # initialize servers
        servers = [
            Server(i, self.serverSpeed, self.serverLocation[i], Server.IDLE, self.serverRotation[i])
            for i in range(self.numServers)
        ]

        # initialize customer and batch queue
        customersQueue = deque()
        batchesQueue = deque()

        # initialize sojourn, time to delivery and clustering
        res_sojourn = SimResults()
        res_delivery = SimResults()
        clustering = Clustering()

        # schedule first arrival
        a = self.interArrivalDist.rvs()
        batchSize = self.batchDist.rvs()
        batch1 = Batch(arrivalTime=a, batchSize=batchSize)
        customers = [
            Customer(a, self.locationDist.rvs(), Customer.WAITING, batch1)
            for _ in range(batchSize)
        ]
        batch1.customers = customers
        customersQueue.extend(customers)
        batchesQueue.extend([batch1])

        fes.add(Event(Event.ARRIVAL, a, batch1, customers))

        # main loop
        while num_delivered < threshold_deliveries:
            event = fes.next()
            batch = event.batch
            prev_t = t
            t = event.event_time

            for s in servers:
                if s.status != Server.SERVING and s.items_in_hand:
                    time_to_depot = s.get_time_to_reach_location(0.0)
                    depot_crossing_time = prev_t + time_to_depot
                    if prev_t < depot_crossing_time <= t:
                        # This server crossed the depot while traveling. Schedule an immediate departure.
                        for e in list(fes.events):
                            if e.server == s and e.event_type == Event.SERVICE:
                                fes.events.remove(e)
                                heapq.heapify(fes.events)
                                # Reschedule the customer it was going to
                                e.customers.status = Customer.WAITING
                                break
                        # Add a new departure event at the crossing time
                        s.location = 0.0
                        s.time_last_location_update = depot_crossing_time
                        fes.add(Event(Event.DEPARTURE, depot_crossing_time, None, s.items_in_hand, s))

            # update server locations
            for server in servers:
                if server.status != Server.SERVING:
                    server.update_location(t)
                    server.time_last_location_update = t


            if event.event_type == Event.ARRIVAL:
                a = self.interArrivalDist.rvs()  # next arrival time
                batchSize = self.batchDist.rvs()
                nextBatch = Batch(arrivalTime=a + t, batchSize=batchSize)
                nextCustomers = [
                    Customer(a + t, self.locationDist.rvs(), Customer.WAITING, nextBatch)
                    for _ in range(batchSize)
                ]
                nextBatch.customers = nextCustomers
                customersQueue.extend(nextCustomers)
                batchesQueue.extend([nextBatch])
                current_locations = np.array([s.location for s in servers])
                clustering.calculate_indices(current_locations)
                fes.add(Event(Event.ARRIVAL, a + t, nextBatch, nextCustomers))

            elif event.event_type == Event.SERVICE:
                customer = event.customers
                server = event.server
                if customer.status != Customer.WAITING:
                    server.status = Server.IDLE
                    continue
                customer.status = Customer.BEING_SERVED
                if customer in customersQueue:
                    customersQueue.remove(customer)
                server.location = customer.location
                server.status = Server.SERVING
                server.time_last_location_update = t
                server.items_in_hand.append(customer)
                service_time = self.serviceDist.rvs()
                fes.add(Event(Event.END_SERVICE, t + service_time, customer.batch, customer, server))

            elif event.event_type == Event.END_SERVICE:
                customer = event.customers
                batch = event.batch
                server = event.server
                customer.status = Customer.PICKED
                server.time_last_location_update = t
                server.status = Server.IDLE
                if batch.all_served():
                    res_sojourn.registerWaitingTime(t - batch.systemArrivalTime)

            elif event.event_type == Event.DEPARTURE:
                server = event.server
                customers = event.customers if event.customers else []
                for customer in customers:
                    customer.status = Customer.DELIVERED
                server.location = 0.0
                server.items_in_hand = []
                server.status = Server.IDLE
                for batch in list(batchesQueue):
                    if batch.all_delivered():
                        batchesQueue.remove(batch)
                        num_delivered += 1
                        res_delivery.registerWaitingTime(t - batch.systemArrivalTime)
            self.schedule_next_event(servers, customersQueue, fes, t)

        return res_sojourn, res_delivery , clustering





