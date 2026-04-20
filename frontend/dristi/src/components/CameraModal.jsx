export default function CameraModal({ close }) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      <div className="bg-white p-6 rounded-2xl w-3/4">
        <h2 className="text-xl font-bold mb-4">Camera</h2>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-200 h-64 flex items-center justify-center">
            Webcam
          </div>

          <div>
            <p>Show the sign like this:</p>
            <div className="bg-gray-100 h-32 mt-2"></div>

            <button className="mt-4 bg-blue-600 text-white px-4 py-2 rounded">
              Check My Sign
            </button>
          </div>
        </div>

        <button onClick={close} className="mt-4 text-red-500">
          Close
        </button>
      </div>
    </div>
  );
}