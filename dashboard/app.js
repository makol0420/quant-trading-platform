async function loadState(){

    const response=await fetch("/api/state/paper");

    const data=await response.json();

    document.getElementById("status").textContent=
        JSON.stringify(data,null,2);

    if(data.cash!==undefined)
        document.getElementById("portfolio").textContent="$"+data.cash;

}

loadState();

setInterval(loadState,5000);

const ctx=document.getElementById("equityChart");

new Chart(ctx,{
    type:"line",
    data:{
        labels:["Mon","Tue","Wed","Thu","Fri"],
        datasets:[
        {
            label:"Equity",
            data:[10000,10120,10090,10350,10500]
        }]
    }
});
