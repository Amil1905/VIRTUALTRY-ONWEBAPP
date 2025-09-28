export default function Login() {
  return (
    <div className="flex justify-center items-center h-screen bg-gray-100">
      <form className="bg-white p-8 rounded shadow-md w-80">
        <h2 className="text-2xl font-semibold mb-4 text-center">Login</h2>
        <input type="email" placeholder="Email" className="border p-2 mb-4 w-full rounded"/>
        <input type="password" placeholder="Password" className="border p-2 mb-4 w-full rounded"/>
        <button className="bg-blue-600 text-white p-2 w-full rounded hover:bg-blue-700">Login</button>
      </form>
    </div>
  );
}
