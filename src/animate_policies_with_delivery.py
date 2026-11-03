import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set font to support math rendering
plt.rcParams['mathtext.fontset'] = 'cm'

from collections import deque
import heapq

# --- Import Classes (Assuming these exist in your 'Base' module) ---
from Base.FES import FES
from Base.Event import Event
from Base.Server import Server
from Base.Customer import Customer
from Base.Batch import Batch
from Base.Simulation import Simulation


class LoggingSimulation(Simulation):
    def run_with_logging(self, max_time):
        fes = FES()
        t = 0

        servers = [
            Server(i, self.serverSpeed, self.serverLocation[i], Server.IDLE, self.serverRotation[i])
            for i in range(self.numServers)
        ]

        customersQueue = deque()
        batchesQueue = deque()
        all_customers = []

        batch_counter = 1
        delivered_batches_count = 0
        total_delivery_time = 0.0
        current_mean_DB = 0.0

        a = self.interArrivalDist.rvs()
        batchSize = int(self.batchDist.rvs())
        batch1 = Batch(arrivalTime=a, batchSize=batchSize)
        batch1.id = batch_counter
        batch_counter += 1

        customers = [
            Customer(a, self.locationDist.rvs(), Customer.WAITING, batch1)
            for _ in range(batchSize)
        ]
        batch1.customers = customers
        customersQueue.extend(customers)
        all_customers.extend(customers)
        batchesQueue.extend([batch1])

        fes.add(Event(Event.ARRIVAL, a, batch1, customers))

        logs = []
        self.log_state(logs, 0, servers, all_customers, delivered_batches_count, current_mean_DB)

        while t < max_time:
            if fes.isEmpty(): break

            event = fes.next()
            prev_t = t
            t = event.event_time
            if t > max_time: break

            # 1. Check Depot Crossings
            for s in servers:
                if s.status != Server.SERVING and s.items_in_hand:
                    time_to_depot = s.get_time_to_reach_location(0.0)
                    depot_crossing_time = prev_t + time_to_depot
                    if prev_t < depot_crossing_time <= t:
                        for e in list(fes.events):
                            if e.server == s and e.event_type == Event.SERVICE:
                                fes.events.remove(e)
                                heapq.heapify(fes.events)
                                e.customers.status = Customer.WAITING
                                break
                        s.location = 0.0
                        s.time_last_location_update = depot_crossing_time
                        fes.add(Event(Event.DEPARTURE, depot_crossing_time, None, s.items_in_hand, s))

            # 2. Update Server Locations
            for server in servers:
                if server.status != Server.SERVING:
                    server.update_location(t)
                    server.time_last_location_update = t

            # 3. Process Event
            if event.event_type == Event.ARRIVAL:
                a = self.interArrivalDist.rvs()
                batchSize = int(self.batchDist.rvs())
                nextBatch = Batch(arrivalTime=a + t, batchSize=batchSize)
                nextBatch.id = batch_counter
                batch_counter += 1

                nextCustomers = [
                    Customer(a + t, self.locationDist.rvs(), Customer.WAITING, nextBatch)
                    for _ in range(batchSize)
                ]
                nextBatch.customers = nextCustomers
                customersQueue.extend(nextCustomers)
                all_customers.extend(nextCustomers)
                batchesQueue.extend([nextBatch])
                fes.add(Event(Event.ARRIVAL, a + t, nextBatch, nextCustomers))

            elif event.event_type == Event.SERVICE:
                customer = event.customers
                server = event.server
                if customer.status == Customer.WAITING:
                    customer.status = Customer.BEING_SERVED
                    if customer in customersQueue: customersQueue.remove(customer)
                    server.location = customer.location
                    server.status = Server.SERVING
                    server.time_last_location_update = t
                    server.items_in_hand.append(customer)
                    service_time = self.serviceDist.rvs()
                    fes.add(Event(Event.END_SERVICE, t + service_time, customer.batch, customer, server))
                else:
                    server.status = Server.IDLE

            elif event.event_type == Event.END_SERVICE:
                event.customers.status = Customer.PICKED
                event.server.status = Server.IDLE
                event.server.time_last_location_update = t

            elif event.event_type == Event.DEPARTURE:
                for c in (event.customers or []):
                    c.status = Customer.DELIVERED
                event.server.location = 0.0
                event.server.items_in_hand = []
                event.server.status = Server.IDLE

                for batch in list(batchesQueue):
                    if batch.all_delivered():
                        batchesQueue.remove(batch)
                        delivered_batches_count += 1

                        delivery_time = t - batch.arrivalTime
                        total_delivery_time += delivery_time
                        current_mean_DB = total_delivery_time / delivered_batches_count

            # 4. Schedule Next
            self.schedule_next_event(servers, customersQueue, fes, t)

            # 5. Log State
            self.log_state(logs, t, servers, all_customers, delivered_batches_count, current_mean_DB)

        return logs

    def log_state(self, logs, t, servers, all_customers, delivered_count, mean_db):
        server_data = [{'location': s.location, 'status': s.status, 'rotation': s.rotationDirection} for s in servers]
        cust_data = [{'location': c.location, 'status': c.status, 'arrival_time': c.arrivalTime, 'batch_id': c.batch.id}
                     for c in all_customers if c.status != Customer.DELIVERED]

        logs.append({
            'time': t,
            'servers': server_data,
            'customers': cust_data,
            'delivered_count': delivered_count,
            'mean_db': mean_db
        })


def get_coords(location):
    angle = np.pi / 2 - 2 * np.pi * location
    return np.cos(angle), np.sin(angle)


def create_animation_from_logs(logs, output_file='system_animation.gif'):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Static Elements
    warehouse = plt.Circle((0, 0), 1.0, color='lightgray', fill=False, linewidth=3)
    ax.add_artist(warehouse)

    depot_x, depot_y = get_coords(0.0)
    ax.plot(depot_x, depot_y, 'ks', markersize=15, label='Depot')

    for i in range(1, 10):
        loc = i / 10.0
        x, y = get_coords(loc)
        ax.text(x * 0.85, y * 0.85, f"{loc:.1f}", ha='center', va='center', fontsize=8, color='gray')

    # Dynamic Elements
    scat_cw = ax.scatter([], [], c='blue', marker='s', s=150, label='CW Server')
    scat_ccw = ax.scatter([], [], c='red', marker='s', s=150, label='CCW Server')
    scat_cust = ax.scatter([], [], c='green', marker='o', s=80, label='Customer')

    time_text = ax.text(-1.2, 1.2, '', fontsize=12)
    delivered_text = ax.text(0.5, 1.2, '', fontsize=12, ha='right')

    # --- UPDATED: Math Text for Mean DB ---
    # Using raw string r'' allows LaTeX formatting.
    # \mathbb{E} makes the blackboard bold E. ^B makes the superscript.
    mean_db_text = ax.text(-1.2, -1.2, '', fontsize=14, color='darkred', fontweight='bold')

    batch_labels = []

    ax.legend(loc='lower right', fontsize=10, markerscale=0.7)

    # --- UPDATED: Slower Animation Logic ---
    fps = 30
    duration = logs[-1]['time']

    # To make it 0.5x speed (twice as slow), we double the number of frames
    # for the same simulation duration.
    # Original: step = 1/fps. New: step = 0.5/fps (sim time advances slower per frame)
    slow_factor = 0.5
    times = np.arange(0, duration, slow_factor / fps)

    log_times = np.array([l['time'] for l in logs])

    def update(frame_idx):
        t = times[frame_idx]

        for txt in batch_labels:
            txt.remove()
        batch_labels.clear()

        idx = np.searchsorted(log_times, t, side='right') - 1
        idx = max(0, idx)
        state = logs[idx]

        elapsed = t - state['time']
        cw_pts, ccw_pts = [], []

        for s in state['servers']:
            loc = s['location']
            if s['status'] != 0:
                loc = (loc + s['rotation'] * 1.0 * elapsed) % 1.0
            x, y = get_coords(loc)
            if s['rotation'] == 1:
                cw_pts.append((x, y))
            else:
                ccw_pts.append((x, y))

        scat_cw.set_offsets(cw_pts if cw_pts else np.zeros((0, 2)))
        scat_ccw.set_offsets(ccw_pts if ccw_pts else np.zeros((0, 2)))

        cust_pts = []
        for c in state['customers']:
            if t >= c['arrival_time'] and c['status'] in [0, 1]:
                cx, cy = get_coords(c['location'])
                cust_pts.append((cx, cy))
                lbl = ax.text(cx * 1.1, cy * 1.1, str(c['batch_id']), fontsize=8, color='black', ha='center')
                batch_labels.append(lbl)

        scat_cust.set_offsets(cust_pts if cust_pts else np.zeros((0, 2)))

        time_text.set_text(f'Time: {t:.2f}')
        delivered_text.set_text(f'Delivered Batches: {state["delivered_count"]}')

        # Update Math Text
        mean_val = state.get('mean_db', 0.0)
        mean_db_text.set_text(r'$\mathbb{E}[D^B]$: ' + f'{mean_val:.2f}')

        return scat_cw, scat_ccw, scat_cust, time_text, delivered_text, mean_db_text, *batch_labels

    ani = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    ani.save(output_file, writer='pillow', fps=fps)
    print(f"Animation saved to {output_file}")


# --- Main Execution ---
def run_scenario(policy_name, rotations, seed=42, output_filename="animation.gif"):
    print(f"--- Running {policy_name} Policy (Seed {seed}) ---")

    np.random.seed(seed)

    interArrivalDist = stats.expon(scale=1 / 0.84)
    batchDist = stats.geom(p=0.2)
    locationDist = stats.uniform(0, 1)
    serviceDist = stats.expon(scale=1.0)

    sim = LoggingSimulation(
        interArrivalDist=interArrivalDist,
        batchDist=batchDist,
        locationDist=locationDist,
        serviceDist=serviceDist,
        numServers=6,
        serverSpeed=1.0,
        serverLocation=[0.0] * 6,
        serverRotation=rotations
    )

    logs = sim.run_with_logging(max_time=20.0)

    print(f"Creating animation for {policy_name}...")
    create_animation_from_logs(logs, output_file=output_filename)
    print(f"Saved {output_filename}\n")


if __name__ == "__main__":
    DEMO_SEED = 12345

    run_scenario(
        policy_name="Unidirectional",
        rotations=[1, 1, 1, 1, 1, 1],
        seed=DEMO_SEED,
        output_filename="unidirectional.gif"
    )

    run_scenario(
        policy_name="Bidirectional",
        rotations=[1, -1, 1, -1, 1, -1],
        seed=DEMO_SEED,
        output_filename="bidirectional.gif"
    )