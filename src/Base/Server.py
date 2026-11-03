class Server:
    SERVING = 0
    TRAVELING = 1
    IDLE = 2
    AT_DEPOT = 3

    def __init__(self, id, speed, location, status, rotationDirection):
        self.id = id
        self.speed = speed
        self.location = location
        self.status = status
        self.rotationDirection = rotationDirection
        self.time_last_location_update = 0.0
        self.items_in_hand = []

    def get_time_to_reach_location(self, target_location):
        if self.rotationDirection == 1:  # Clockwise
            distance = target_location - self.location
            if distance < 0:
                distance += 1
        elif self.rotationDirection == -1:  # Counter-clockwise
            distance = self.location - target_location
            if distance < 0:
                distance += 1
        return distance / self.speed

    def update_location(self, current_time):
        travel_distance = self.speed * (current_time - self.time_last_location_update)
        self.location = float((self.location + self.rotationDirection * travel_distance) % 1.0)

    def crossed_depot(self, target_location):
        if self.rotationDirection == 1:  # Clockwise
            crosses_depot = target_location < self.location
        else:  # Counter-clockwise
            crosses_depot = target_location > self.location
        return crosses_depot


    def __str__(self):
        return f"Server {self.id} in location {self.location}"