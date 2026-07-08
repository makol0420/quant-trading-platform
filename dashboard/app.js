async function loadState() {
    try {
        const response = await fetch("/api/state/paper");
        const data = await response.json();

        document.getElementById("portfolio").textContent =
            "$" + (data.portfolio ?? 0).toLocaleString();

        document.getElementById("cash").textContent =
            "$" + (data.cash ?? 0).toLocaleString();

        document.getElementById("pnl").textContent =
            "$" + (data.daily_pl ?? 0);

        document.getElementById("trades").textContent =
            data.open_trades ?? 0;

        document.getElementById("winrate").textContent =
            (data.win_rate ?? 0) + "%";

        document.getElementById("status").textContent =
            JSON.stringify(data, null, 2);

        updatePositions(data.positions || []);
    }
    catch (err) {
        console.error(err);
    }
}

function updatePositions(positions) {

    const tbody = document.getElementById("positions");

    tbody.innerHTML = "";

    positions.forEach(pos => {

        tbody.innerHTML += `
        <tr>
            <td>${pos.symbol}</td>
            <td>${pos.side}</td>
            <td>${pos.entry}</td>
            <td>${pos.current}</td>
            <td>$${pos.pnl}</td>
        </tr>`;
    });

}

loadState();

setInterval(loadState,5000);

new Chart(document.getElementById("equityChart"),{
    type:"line",
    data:{
        labels:["Mon","Tue","Wed","Thu","Fri"],
        datasets:[{
            label:"Portfolio Equity",
            data:[10000,10120,10090,10350,10500],
            borderWidth:3,
            tension:.4
        }]
    },
    options:{
        responsive:true,
        plugins:{
            legend:{
                labels:{color:"white"}
            }
        },
        scales:{
            x:{
                ticks:{color:"white"}
            },
            y:{
                ticks:{color:"white"}
            }
        }
    }
});
