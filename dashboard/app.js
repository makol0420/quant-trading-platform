let chart;

// Load dashboard data
async function loadState() {
    try {
        const response = await fetch("/api/state/paper");
        const data = await response.json();

        // Update cards
        document.getElementById("portfolio").textContent =
            "$" + (data.portfolio ?? data.cash ?? 0).toLocaleString();

        document.getElementById("pnl").textContent =
            "$" + (data.daily_pl ?? 0);

        document.getElementById("trades").textContent =
            data.open_trades ?? 0;

        document.getElementById("winrate").textContent =
            (data.win_rate ?? 0) + "%";

        // Show JSON status
        document.getElementById("status").textContent =
            JSON.stringify(data, null, 2);

    } catch (err) {

        document.getElementById("status").textContent =
            "Unable to connect to the API.";

        console.error(err);

    }
}

// Create chart
function createChart() {

    const ctx = document.getElementById("equityChart");

    chart = new Chart(ctx, {

        type: "line",

        data: {

            labels: [
                "Mon",
                "Tue",
                "Wed",
                "Thu",
                "Fri",
                "Sat",
                "Sun"
            ],

            datasets: [{

                label: "Portfolio Value",

                data: [
                    10000,
                    10120,
                    10250,
                    10190,
                    10400,
                    10520,
                    10650
                ],

                borderWidth: 3,

                tension: 0.35,

                fill: false

            }]

        },

        options: {

            responsive: true,

            plugins: {

                legend: {
                    display: true
                }

            }

        }

    });

}

createChart();

loadState();

setInterval(loadState, 5000);
