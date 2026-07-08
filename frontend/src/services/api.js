import axios from "axios";

const api = axios.create({
  baseURL: "https://quant-trading-platform-q8qd.onrender.com",
});

export default api;
