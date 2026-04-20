import Card from "./Card";

export default function Dashboard({ openCamera }) {
  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card title="Signs Learned" value="25" />
        <Card title="Streak" value="7 days" />
        <Card title="Accuracy" value="89%" />
        <Card title="Time" value="2h" />
      </div>

      <button
        onClick={openCamera}
        className="bg-blue-600 text-white px-6 py-3 rounded-xl"
      >
        Open Camera
      </button>
    </div>
  );
}