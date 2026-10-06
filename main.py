from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import heapq

app = Flask(__name__)

# ---------------- AI TRAFFIC DATA ----------------

data = {
    "hour": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
             17, 18, 19, 20, 21, 22],

    "vehicles": [300, 850, 950, 650, 400, 350, 300, 320,
                 450, 600, 850, 1000, 950, 700, 450, 300],

    "speed": [45, 22, 18, 28, 40, 45, 50, 48,
              42, 32, 20, 15, 18, 28, 40, 48],

    "traffic": [
        "Medium", "High", "High", "High",
        "Medium", "Low", "Low", "Low",
        "Medium", "Medium", "High", "High",
        "High", "Medium", "Low", "Low"
    ]
}

df = pd.DataFrame(data)

X = df[["hour", "vehicles", "speed"]]
y = df["traffic"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# ---------------- ROAD NETWORK ----------------

road_network = {
    "A": [("B", 5), ("C", 8)],
    "B": [("A", 5), ("C", 3), ("D", 7)],
    "C": [("A", 8), ("B", 3), ("D", 2)],
    "D": [("B", 7), ("C", 2), ("E", 4)],
    "E": [("D", 4), ("F", 6)],
    "F": [("E", 6)]
}


# ---------------- FIND BEST ROUTE ----------------

def find_best_route(graph, start, destination):

    priority_queue = [(0, start, [])]
    visited = set()

    while priority_queue:

        cost, current, path = heapq.heappop(priority_queue)

        if current in visited:
            continue

        visited.add(current)

        new_path = path + [current]

        if current == destination:
            return cost, new_path

        for neighbour, distance in graph.get(current, []):

            if neighbour not in visited:

                heapq.heappush(
                    priority_queue,
                    (cost + distance, neighbour, new_path)
                )

    return None, []


# ---------------- ROUTE OPTIMIZATION ----------------

def optimize_route(start, destination, traffic_level):

    if traffic_level == "High":
        factor = 2.0

    elif traffic_level == "Medium":
        factor = 1.4

    else:
        factor = 1.0

    modified_graph = {}

    for location, roads in road_network.items():

        modified_graph[location] = []

        for neighbour, distance in roads:

            new_distance = distance * factor

            modified_graph[location].append(
                (neighbour, new_distance)
            )

    return find_best_route(
        modified_graph,
        start,
        destination
    )


# ---------------- PAGE 1 ----------------

HOME_PAGE = """

<!DOCTYPE html>

<html>

<head>

<title>AI Traffic Prediction</title>

<style>

body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    min-height: 100vh;

    background:
        radial-gradient(circle at 10% 15%, rgba(37, 99, 235, 0.40), transparent 30%),
        radial-gradient(circle at 90% 85%, rgba(139, 92, 246, 0.35), transparent 30%),
        radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.15), transparent 40%),
        linear-gradient(135deg, #020617, #0f172a, #111827);

    color: white;
}

.container {
    width: 70%;
    margin: auto;
    padding: 40px;
}

 h1 {
    text-align: center;
    color: #ffffff;
    font-size: 32px;
    margin-bottom: 10px;
    letter-spacing: 1px;

    text-shadow:
        0 0 5px #ffffff,
        0 0 10px #38bdf8,
        0 0 20px #38bdf8,
        0 0 35px #2563eb;

    animation: titleGlow 2s ease-in-out infinite alternate;
}
.section-title {
    color: #e0f2fe;
    font-size: 22px;
    margin-bottom: 20px;
    letter-spacing: 0.8px;

    text-shadow:
        0 0 6px rgba(56, 189, 248, 0.6),
        0 0 12px rgba(56, 189, 248, 0.35);
}

.card {
    background: rgba(255, 255, 255, 0.10);
    padding: 30px;
    margin-top: 25px;
    border-radius: 20px;

    border: 1px solid rgba(255, 255, 255, 0.20);

    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);

    backdrop-filter: blur(12px);
    .card {
    ...
    backdrop-filter: blur(12px);
    animation: fadeIn 0.8s ease;
}
}

label {
    display: block;
    margin-top: 18px;
    margin-bottom: 6px;

    color: #e0f2fe;
    font-weight: bold;
    font-size: 14px;
    letter-spacing: 0.5px;

    text-shadow: 0 0 6px rgba(56, 189, 248, 0.35);
}

input, select {
    width: 95%;
    padding: 13px;
    margin-top: 6px;

    background: rgba(255, 255, 255, 0.12);
    color: #e0f2fe;
    font-weight: bold;

    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 10px;

    outline: none;
    font-size: 15px;
}
input, select {
    ...
}

input:focus, select:focus {
    border-color: #22d3ee;
    box-shadow: 0 0 12px rgba(34, 211, 238, 0.4);
}

button {
    margin-top: 25px;
    padding: 14px 28px;

    background: linear-gradient(135deg, #06b6d4, #2563eb);

    color: white;
    border: none;
    border-radius: 10px;

    cursor: pointer;
    font-size: 16px;
    font-weight: bold;

    box-shadow: 0 5px 20px rgba(0, 180, 255, 0.35);

    transition: 0.3s;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 200, 255, 0.55);
}
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}
@media (max-width: 700px) {
    .container {
        width: 90%;
        padding: 30px 15px;
    }
h2 {
    color: #c4b5fd;
    font-size: 22px;
}
    h1 {
        font-size: 25px;
    }

    .card {
        padding: 22px;
    }

    input, select {
        width: 90%;
    }
}
</style>

</head>


<body>

<div class="container">

<h1>
AI-Based Traffic Prediction and Smart Route Optimization System
</h1>


<div class="badge">🤖 AI POWERED • SMART TRAFFIC SYSTEM</div>


<div class="card">

<h2 class="section-title">Enter Traffic Details</h2>


<form method="POST" action="/predict">


<label>Time (Hour)</label>

<input
type="number"
name="hour"
min="0"
max="23"
required
>


<label>Number of Vehicles</label>

<input
type="number"
name="vehicles"
min="0"
required
>


<label>Average Vehicle Speed (km/h)</label>

<input
type="number"
name="speed"
min="1"
required
>


<label>Starting Location</label>

<select name="start">

<option value="A">A</option>
<option value="B">B</option>
<option value="C">C</option>
<option value="D">D</option>
<option value="E">E</option>

</select>


<label>Destination</label>

<select name="destination">

<option value="F">F</option>
<option value="E">E</option>
<option value="D">D</option>
<option value="C">C</option>
<option value="B">B</option>

</select>


<button type="submit">

Predict Traffic & Find Best Route

</button>


</form>

</div>

</div>

</body>

</html>

"""


# ---------------- PAGE 2 ----------------

RESULT_PAGE = """

<!DOCTYPE html>

<html>

<head>

<title>Prediction Result</title>

<style>

body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    min-height: 100vh;

    background:
        radial-gradient(circle at 15% 20%, rgba(59, 130, 246, 0.35), transparent 30%),
        radial-gradient(circle at 85% 80%, rgba(168, 85, 247, 0.30), transparent 30%),
        radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.12), transparent 40%),
        linear-gradient(135deg, #020617, #0f172a, #111827);

    color: white;
}

.container {
    width: 70%;
    max-width: 900px;
    margin: auto;
    padding: 60px 40px;
}

h1 {
    text-align: center;
    color: #c4b5fd;
    font-size: 32px;
    margin-bottom: 10px;
    letter-spacing: 1px;
    text-shadow: 0 0 15px rgba(0, 200, 255, 0.5);
}
.badge {
    text-align: center;
    color: #67e8f9;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 1px;
    margin-top: 8px;
}

.card {
    background: rgba(255, 255, 255, 0.96);
    padding: 30px;
    margin-top: 25px;
    border-radius: 20px;

    border: 1px solid rgba(255, 255, 255, 0.8);

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.35),
        0 0 25px rgba(56, 189, 248, 0.18);

    color: #0f172a;

    backdrop-filter: blur(12px);
}

.result {

    padding: 20px;

    margin-top: 20px;

    background: #f4f4f4;

    border-radius: 8px;

}

.high {

    color: red;

    font-weight: bold;

}

.medium {

    color: orange;

    font-weight: bold;

}

.low {

    color: green;

    font-weight: bold;

}

.back {

    display: inline-block;

    margin-top: 20px;

    padding: 12px 20px;

    background: #222;

    color: white;

    text-decoration: none;

    border-radius: 6px;

}

</style>

</head>


<body>


<div class="container">


<h1>

AI Traffic Prediction Result

</h1>


<div class="card">


<div class="result">

<h2>Traffic Prediction</h2>

<p>

<strong>Predicted Traffic Level:</strong>

<span class="{{ traffic_class }}">

{{ traffic }}

</span>

</p>


<p>

<strong>Number of Vehicles:</strong>

{{ vehicles }}

</p>


<p>

<strong>Average Speed:</strong>

{{ speed }} km/h

</p>

</div>



<div class="result">

<h2>Smart Route Recommendation</h2>


<p>

<strong>Starting Location:</strong>

{{ start }}

</p>


<p>

<strong>Destination:</strong>

{{ destination }}

</p>


<p>

<strong>Recommended Route:</strong>

{{ route }}

</p>


<p>

<strong>Estimated Route Cost:</strong>

{{ cost }}

</p>


<p>

The route has been optimized according to the predicted traffic condition.

</p>


</div>


<a class="back" href="/">

← Predict Again

</a>


</div>


</div>


</body>

</html>

"""


# ---------------- HOME ROUTE ----------------

@app.route("/")

def home():

    return render_template_string(HOME_PAGE)


# ---------------- PREDICTION ROUTE ----------------

@app.route("/predict", methods=["POST"])

def predict():

    hour = int(request.form["hour"])

    vehicles = int(request.form["vehicles"])

    speed = float(request.form["speed"])

    start = request.form["start"]

    destination = request.form["destination"]


    # AI prediction

    input_data = pd.DataFrame(
        [[hour, vehicles, speed]],
        columns=["hour", "vehicles", "speed"]
    )

    traffic = model.predict(input_data)[0]


    # Traffic color

    if traffic == "High":

        traffic_class = "high"

    elif traffic == "Medium":

        traffic_class = "medium"

    else:

        traffic_class = "low"


    # Find optimized route

    route_cost, route_path = optimize_route(
        start,
        destination,
        traffic
    )


    if route_path:

        route = " → ".join(route_path)

        cost = round(route_cost, 2)

    else:

        route = "No route available"

        cost = "N/A"


    # Open result page

    return render_template_string(

        RESULT_PAGE,

        traffic=traffic,

        traffic_class=traffic_class,

        vehicles=vehicles,

        speed=speed,

        start=start,

        destination=destination,

        route=route,

        cost=cost

    )


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":

    print("------------------------------------------")

    print("AI Traffic Prediction System")

    print("------------------------------------------")

    print("Open: http://127.0.0.1:5000")

    print("------------------------------------------")

    app.run(debug=True)