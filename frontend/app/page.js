"use client";

import { useState, useEffect } from "react";
import ChatInterface from "../components/ChatInterface";
import SystemStatus from "../components/SystemStatus";
import DocumentStats from "../components/DocumentStats";
import { apiService } from "../lib/api";
import { Settings, MessageSquare, BarChart3, RefreshCw } from "lucide-react";

export default function RagApplication() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("chat");
  const [error, setError] = useState(null);

  useEffect(() => {
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
    } catch (error) {
      console.error("Failed to check system status:", error);
      setError(
        "Failed to connect to the backend. Please ensure the server is running."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const initializeSystem = async (forceRebuild = false) => {
    try {
      setIsLoading(true);
      setError(null);
      await apiService.initializeSystem(forceRebuild);
      await checkSystemStatus();
    } catch (error) {
      console.error("Failed to initialize system:", error);
      setError(
        "Failed to initialize the system. Please check the server logs."
      );
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading && !systemStatus) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-700 mb-2">
            Loading RAG System
          </h2>
          <p className="text-gray-500">Checking system status...</p>
        </div>
      </div>
    );
  }

  if (error && !systemStatus) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-lg text-center max-w-md">
          <div className="text-red-500 text-5xl mb-4">⚠️</div>
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Connection Error
          </h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={() => checkSystemStatus()}
            className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  const needsInitialization = systemStatus && !systemStatus.system_initialized;

  if (needsInitialization) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-lg text-center max-w-md">
          <div className="text-yellow-500 text-5xl mb-4">🚀</div>
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            System Initialization Required
          </h2>
          <p className="text-gray-600 mb-6">
            The RAG system needs to be initialized. This will download research
            papers and build the vector database.
          </p>
          <div className="space-y-3">
            <button
              onClick={() => initializeSystem(false)}
              disabled={isLoading}
              className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin inline mr-2" />
                  Initializing...
                </>
              ) : (
                "Initialize System"
              )}
            </button>
            <button
              onClick={() => initializeSystem(true)}
              disabled={isLoading}
              className="w-full px-6 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Force Rebuild
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">
                RAG Application
              </h1>
              <span className="ml-3 px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                {systemStatus?.system_initialized ? "Ready" : "Initializing"}
              </span>
            </div>

            <nav className="flex space-x-4">
              <button
                onClick={() => setActiveTab("chat")}
                className={`px-3 py-2 rounded-md text-sm font-medium flex items-center ${
                  activeTab === "chat"
                    ? "bg-blue-100 text-blue-700"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <MessageSquare className="w-4 h-4 mr-2" />
                Chat
              </button>

              <button
                onClick={() => setActiveTab("status")}
                className={`px-3 py-2 rounded-md text-sm font-medium flex items-center ${
                  activeTab === "status"
                    ? "bg-blue-100 text-blue-700"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <Settings className="w-4 h-4 mr-2" />
                Status
              </button>

              <button
                onClick={() => setActiveTab("stats")}
                className={`px-3 py-2 rounded-md text-sm font-medium flex items-center ${
                  activeTab === "stats"
                    ? "bg-blue-100 text-blue-700"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                Statistics
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {activeTab === "chat" && <ChatInterface />}
        {activeTab === "status" && (
          <SystemStatus
            status={systemStatus}
            onRefresh={checkSystemStatus}
            onInitialize={initializeSystem}
          />
        )}
        {activeTab === "stats" && <DocumentStats />}
      </main>
    </div>
  );
}
