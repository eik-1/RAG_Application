import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor for logging
api.interceptors.request.use((config) => {
  console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API Error:", error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Chat endpoints
  async sendMessage(request) {
    const response = await api.post("/chat", request);
    return response.data;
  },

  // System management
  async getSystemStatus() {
    const response = await api.get("/status");
    return response.data;
  },

  async getHealth() {
    const response = await api.get("/health");
    return response.data;
  },

  async initializeSystem(forceRebuild = false) {
    const response = await api.post("/initialize", {
      force_rebuild: forceRebuild,
    });
    return response.data;
  },

  async rebuildSystem() {
    const response = await api.post("/rebuild");
    return response.data;
  },

  // Memory management
  async clearMemory() {
    const response = await api.post("/memory/clear");
    return response.data;
  },

  async getMemoryStatus() {
    const response = await api.get("/memory/status");
    return response.data;
  },

  // Document operations
  async getDocumentStats() {
    const response = await api.get("/documents/stats");
    return response.data;
  },

  async searchDocuments(query, topK = 10) {
    const response = await api.post("/search", { query, top_k: topK });
    return response.data;
  },
};

export default apiService;
