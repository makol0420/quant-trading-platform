const items = [
  "Dashboard",
  "Portfolio",
  "Orders",
  "Trades",
  "Markets",
  "Strategies",
  "Backtesting",
  "Settings",
];

export default function Sidebar() {
  return (
    <div className="w-64 h-screen bg-slate-900 border-r border-slate-800">
      <div className="text-2xl font-bold p-6">
        Quant Platform
      </div>

      {items.map((item) => (
        <div
          key={item}
          className="px-6 py-4 hover:bg-slate-800 cursor-pointer"
        >
          {item}
        </div>
      ))}
    </div>
  );
}
