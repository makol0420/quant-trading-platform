export default function Header() {
  const now = new Date().toLocaleString();

  return (
    <div className="flex justify-between items-center p-6 border-b border-slate-800">
      <div>
        <h1 className="text-3xl font-bold">
          Dashboard
        </h1>

        <p className="text-gray-400">
          Quant Trading Platform
        </p>
      </div>

      <div className="text-right">
        <div className="text-green-400">
          ● PAPER
        </div>

        <div className="text-sm text-gray-400">
          {now}
        </div>
      </div>
    </div>
  );
}
