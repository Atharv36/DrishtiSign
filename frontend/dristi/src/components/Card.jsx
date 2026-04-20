export default function Card({ title, value }) {
    return (
      <div className="bg-white shadow rounded-2xl p-4">
        <h3 className="text-gray-500">{title}</h3>
        <p className="text-2xl font-bold">{value}</p>
      </div>
    );
  }
  
  
  // 