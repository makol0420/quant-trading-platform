export default function StatCard({ title, value }) {
  return (
    <div className="bg-slate-900 rounded-xl p-6 shadow">
      <div className="text-gray-400">
        {title}
      </div>

      <div className="text-3xl font-bold mt-2">
        {value}
      </div>
    </div>
  );
}
