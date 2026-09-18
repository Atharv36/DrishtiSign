import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5001/api',
});

api.interceptors.request.use(
  (config) => {
    const userInfo = localStorage.getItem('userInfo');
    if (userInfo) {
      const { token } = JSON.parse(userInfo);
      config.headers.Authorization = `Bearer ${token}`;
    }
    // Let the backend record activity against the user's own local day, not the server's.
    config.headers['X-Client-Date'] = new Date().toLocaleDateString('en-CA');
    return config;
  },
  (error) => Promise.reject(error)
);

export const getActivity = async () => {
  const { data } = await api.get('/activity');
  return data.data;
};

export default api;
