*{
    margin:0;
    padding:0;
    box-sizing:border-box;
    font-family:Arial,Helvetica,sans-serif;
}

body{
    background:#0f172a;
    color:white;
}

.app{
    display:flex;
    min-height:100vh;
}

.sidebar{
    width:260px;
    background:#111827;
    padding:25px;
}

.logo{
    margin-bottom:35px;
}

.logo h2{
    color:#38bdf8;
}

.logo p{
    color:#9ca3af;
    font-size:14px;
}

.sidebar nav{
    display:flex;
    flex-direction:column;
}

.sidebar nav a{
    color:#d1d5db;
    text-decoration:none;
    padding:14px;
    margin-bottom:8px;
    border-radius:8px;
    transition:.3s;
}

.sidebar nav a:hover,
.sidebar nav a.active{
    background:#2563eb;
    color:white;
}

.main{
    flex:1;
    display:flex;
    flex-direction:column;
}

.topbar{
    height:70px;
    background:#1e293b;
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:0 30px;
}

.topbar-right{
    display:flex;
    align-items:center;
    gap:15px;
}

.status{
    color:#22c55e;
    font-weight:bold;
}

.icon-btn{
    background:#334155;
    border:none;
    color:white;
    padding:10px;
    border-radius:8px;
    cursor:pointer;
}

.user{
    background:#334155;
    padding:10px 16px;
    border-radius:20px;
}

.content{
    padding:30px;
}
