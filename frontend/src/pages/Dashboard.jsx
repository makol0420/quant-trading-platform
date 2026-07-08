import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import StatCard from "../components/StatCard";

export default function Dashboard() {
  return (
    <div className="flex">
      <Sidebar />

      <div className="flex-1">
        <Header />

        <div className="grid grid-cols-4 gap-6 p-6">
          <StatCard title="Portfolio Value" value="$100,000" />
          <StatCard title="Cash" value="$54,220" />
          <StatCard title="Today's P/L" value="+$421" />
          <StatCard title="Open Positions" value="7" />
        </div>
      </div>
    </div>
  );
}
