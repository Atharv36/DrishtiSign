export default function Practice({ openCamera }) {
    const signs = ["A", "B", "C", "D"];
  
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold mb-4">Practice</h1>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {signs.map((s) => (
            <div
              key={s}
              onClick={openCamera}
              className="cursor-pointer bg-white p-4 rounded-xl shadow"
            >
              Sign {s}
            </div>
          ))}
        </div>
      </div>
    );
  }