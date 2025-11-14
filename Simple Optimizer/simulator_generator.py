import math
import random
import csv  # <-- 1. IMPORT THE CSV MODULE

stations = [
    "S01UN", "S02UC", "S03US", "S04PL", "S05M11", "S06M10", "S07KP", "S08SP",
    "S09AG", "S10BS", "S11FG", "S12KB", "S13SB", "S14DU", "S15BS", "S16MJ"
]

headways = [
    [0, 30, 10],
    [30, 270, 8],
    [270, 450, 10],
    [450, 810, 8],
    [810, 870, 10]
]

od_matrix = [
    [-1, 2614, 3037, 53646, 73547, 151650, 31849, 38966, 87927, 17074, 51458, 71943, 54814, 59100, 124996, 129478, 0,
     952099],
    [2969, -1, 436, 20366, 18739, 43402, 6491, 9154, 19617, 5563, 16584, 19693, 10565, 13516, 20292, 25264, 0, 232651],
    [2916, 509, -1, 9962, 11964, 17116, 4310, 4096, 6435, 1668, 4692, 5576, 2407, 3716, 4793, 5312, 0, 85472],
    [57564, 20518, 9289, -1, 4016, 40037, 30485, 28453, 64101, 13027, 47142, 65538, 33534, 38943, 71497, 61751, 0,
     585895],
    [83913, 21074, 12003, 4380, -1, 12204, 28951, 46666, 57070, 13099, 55348, 66701, 52291, 46920, 87489, 80082, 0,
     668191],
    [143726, 38428, 12369, 29628, 8758, -1, 9286, 28499, 67369, 23705, 92755, 127172, 76593, 93890, 182115, 183857, 0,
     1118150],
    [37329, 7941, 3883, 30302, 25984, 11170, -1, 1686, 15182, 10365, 33712, 45013, 29968, 26111, 52877, 42318, 0,
     373841],
    [43241, 9830, 4268, 28862, 42404, 35812, 1695, -1, 8196, 10738, 56666, 72141, 40657, 40681, 66749, 64595, 0,
     526535],
    [86770, 18388, 5063, 55044, 48258, 76663, 13452, 8121, -1, 3816, 27329, 73252, 50620, 61135, 134467, 132613, 0,
     794991],
    [19011, 6681, 1556, 15426, 15161, 35349, 11831, 11520, 3694, -1, 4343, 15101, 20197, 26216, 26852, 40084, 0,
     253022],
    [50574, 15974, 3916, 41363, 47465, 98554, 30103, 54173, 30598, 4863, -1, 17310, 38713, 73446, 98291, 155142, 0,
     760485],
    [69580, 19413, 4578, 59207, 58015, 137840, 37952, 62915, 69492, 13947, 11557, -1, 19873, 53785, 119543, 140129, 0,
     877826],
    [58401, 11123, 2151, 31138, 47247, 86685, 27723, 37785, 54448, 23193, 36606, 22141, -1, 4757, 31337, 102505, 0,
     577240],
    [61059, 14102, 2985, 37057, 42198, 101944, 23023, 38020, 62311, 26032, 65659, 53077, 4360, -1, 11158, 60329, 0,
     603314],
    [106143, 18859, 4016, 55950, 68090, 177188, 42021, 55432, 127587, 27037, 80565, 113438, 26901, 12013, -1, 18701, 0,
     933941],
    [143459, 29421, 5132, 62738, 82818, 215612, 40877, 68244, 145448, 40804, 144861, 146226, 92737, 57968, 16106, -1, 0,
     1292451],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# Convert all OD data to daily averages
for i in range(len(od_matrix)):
    for j in range(len(od_matrix[i])):
        if od_matrix[i][j] > 0:
            od_matrix[i][j] /= 31

total_time = int(14.5 * 60)  # 870 minutes
NORMALIZING_CONSTANT = 102.0

# --- SPECIAL EVENT DEFINITION ---
special_event = {
    "station_index": 8,  # S09AG
    "start_time": 300,
    "end_time": 420,
    "type": "alighting",
    "multiplier": 3.0
}


def get_noise_factor(time, origin_idx, dest_idx, event):
    factor = 1.0
    if event["start_time"] <= time <= event["end_time"]:
        if event["type"] == "boarding":
            if origin_idx == event["station_index"]:
                factor = event["multiplier"]
        elif event["type"] == "alighting":
            if dest_idx == event["station_index"]:
                factor = event["multiplier"]
        elif event["type"] == "both":
            if origin_idx == event["station_index"] or dest_idx == event["station_index"]:
                factor = event["multiplier"]
    return factor


def get_current_headway(time):
    for hw in headways:
        if hw[0] <= time < hw[1]:
            return hw[2]
    if time == headways[-1][1]:
        return headways[-1][2]
    return 0


def get_passenger_probability(curr_head, daily_od_passengers):
    if daily_od_passengers <= 0 or curr_head == 0:
        return 0.0
    prob_per_minute = (daily_od_passengers / NORMALIZING_CONSTANT) * (1.0 / curr_head)
    return prob_per_minute


# --- 2. MODIFIED generate_log FUNCTION ---
def generate_log(csv_writer, ori, dest, time, curr_head):
    """
    Generates a passenger log with a RANDOMIZED travel time
    and writes it to the CSV file.
    """
    id_ = "RP" + str(random.randint(1000, 9999))
    wait_time = random.uniform(0, curr_head)
    in_vehicle_time = abs(dest - ori) * 2
    exit_time = time + wait_time + in_vehicle_time

    # Write the row to the CSV file
    csv_writer.writerow([
        time,
        f"{exit_time:.2f}",
        stations[ori],
        stations[dest],
        id_
    ])


# --- 3. NEW FUNCTION TO SET UP THE CSV ---
def setup_csv_writer(filename="passenger_log.csv"):
    """
    Opens a CSV file for writing and writes the header row.
    Returns the file object and the csv.writer object.
    """
    # Use newline='' as recommended for csv module
    file = open(filename, mode='w', newline='')
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(["Entry", "Exit", "Origin", "Dest", "ID"])

    return file, writer


# --- 4. MAIN LOOP MODIFIED FOR CSV WRITING ---

total = 0
output_filename = "simulation_log.csv"

# Open the file once before the loop
try:
    log_file, csv_writer = setup_csv_writer(output_filename)

    for t in range(total_time + 1):  # Loop from t=0 to t=870
        current_headway = get_current_headway(t)
        if current_headway == 0:
            continue

        # Simulating ONLY from S01UN
        for i in range(1):
            for j in range(1, 16):
                daily_od = od_matrix[i][j]
                if daily_od <= 0:
                    continue

                probability = get_passenger_probability(current_headway, daily_od)
                noise_factor = get_noise_factor(t, i, j, special_event)
                final_probability = probability * noise_factor

                full_passengers = math.floor(final_probability)
                fractional_passenger_prob = final_probability - full_passengers

                for k in range(full_passengers):
                    total += 1
                    # Pass the csv_writer to the log generator
                    generate_log(csv_writer, i, j, t, current_headway)

                if random.random() <= fractional_passenger_prob:
                    total += 1
                    # Pass the csv_writer to the log generator
                    generate_log(csv_writer, i, j, t, current_headway)

finally:
    # Ensure the file is closed even if an error occurs
    if 'log_file' in locals() and not log_file.closed:
        log_file.close()

print(f"Total passengers generated: {total}")
print(f"Log file saved as: {output_filename}")