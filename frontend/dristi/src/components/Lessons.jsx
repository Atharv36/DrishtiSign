export default function Lessons({ openCamera }) {
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold mb-4">Lessons</h1>
        <div className="bg-white p-4 rounded-xl shadow">
          <p>Beginner Lesson</p>
          <div className="w-full bg-gray-200 h-2 rounded my-2">
            <div className="bg-blue-600 h-2 w-1/2 rounded"></div>
          </div>
          <button
            onClick={openCamera}
            className="bg-green-600 text-white px-4 py-2 rounded"
          >
            Continue
          </button>
        </div>
      </div>
    );
  }