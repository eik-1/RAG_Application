"use client";

import {
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertCircle,
  Settings,
} from "lucide-react";

const SystemStatus = ({ status, onRefresh, onInitialize }) => {
  const getStatusIcon = (isHealthy) => {
    return isHealthy ? (
      <CheckCircle className="w-5 h-5 text-green-500" />
    ) : (
      <XCircle className="w-5 h-5 text-red-500" />
    );
  };

  const getStatusBadge = (isHealthy, label) => {
    return (
      <span
        className={`px-2 py-1 rounded-full text-xs font-medium ${
          isHealthy ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
        }`}
      >
        {label}
      </span>
    );
  };

  if (!status) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center space-x-2">
          <RefreshCw className="w-5 h-5 animate-spin" />
          <span>Loading system status...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overall Status */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Settings className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">
              System Status
            </h2>
          </div>

          <div className="flex items-center space-x-3">
            {getStatusBadge(
              status.system_initialized,
              status.system_initialized ? "Ready" : "Initializing"
            )}
            <button
              onClick={onRefresh}
              className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
              title="Refresh status"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {!status.system_initialized && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
            <div className="flex">
              <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 mr-3" />
              <div>
                <h3 className="text-sm font-medium text-yellow-800">
                  System Initialization Required
                </h3>
                <p className="text-sm text-yellow-700 mt-1">
                  The RAG system needs to be initialized before you can start
                  chatting.
                </p>
                <div className="mt-3 space-x-2">
                  <button
                    onClick={() => onInitialize(false)}
                    className="px-3 py-1 bg-yellow-600 text-white text-sm rounded hover:bg-yellow-700"
                  >
                    Initialize Now
                  </button>
                  <button
                    onClick={() => onInitialize(true)}
                    className="px-3 py-1 bg-orange-600 text-white text-sm rounded hover:bg-orange-700"
                  >
                    Force Rebuild
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Component Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* PDF Pipeline */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">PDF Pipeline</h3>
            {getStatusIcon(status.components.pdf_pipeline.status === "ready")}
          </div>
          <div className="space-y-2">
            <div className="text-sm text-gray-600">
              Status:{" "}
              <span className="font-medium">
                {status.components.pdf_pipeline.status}
              </span>
            </div>
          </div>
        </div>

        {/* Vector Database */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">
              Vector Database
            </h3>
            {getStatusIcon(
              status.components.vector_database.status === "ready"
            )}
          </div>
          <div className="space-y-2 text-sm text-gray-600">
            <div>
              Status:{" "}
              <span className="font-medium">
                {status.components.vector_database.status}
              </span>
            </div>
            {status.components.vector_database.total_vectors && (
              <>
                <div>
                  Vectors:{" "}
                  <span className="font-medium">
                    {status.components.vector_database.total_vectors.toLocaleString()}
                  </span>
                </div>
                <div>
                  Dimensions:{" "}
                  <span className="font-medium">
                    {status.components.vector_database.embedding_dimension}
                  </span>
                </div>
                <div>
                  Chunks:{" "}
                  <span className="font-medium">
                    {status.components.vector_database.total_chunks?.toLocaleString()}
                  </span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Language Model */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">
              Language Model
            </h3>
            {getStatusIcon(status.components.language_model.loaded)}
          </div>
          <div className="space-y-2 text-sm text-gray-600">
            <div>
              Loaded:{" "}
              <span className="font-medium">
                {status.components.language_model.loaded ? "Yes" : "No"}
              </span>
            </div>
            <div>
              Model:{" "}
              <span className="font-medium text-xs">
                {status.components.language_model.model_name}
              </span>
            </div>
          </div>
        </div>

        {/* Memory Manager */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">
              Memory Manager
            </h3>
            {getStatusIcon(true)}
          </div>
          <div className="space-y-2 text-sm text-gray-600">
            <div>
              Interactions:{" "}
              <span className="font-medium">
                {status.components.memory_manager.total_interactions}
              </span>
            </div>
            <div>
              Memory Limit:{" "}
              <span className="font-medium">
                {status.components.memory_manager.memory_size_limit}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Document Distribution */}
      {status.components.vector_database.document_distribution && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Document Distribution
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(
              status.components.vector_database.document_distribution
            ).map(([doc, count]) => (
              <div key={doc} className="bg-gray-50 p-3 rounded">
                <div
                  className="text-sm font-medium text-gray-900 truncate"
                  title={doc}
                >
                  {doc
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (l) => l.toUpperCase())}
                </div>
                <div className="text-lg font-semibold text-blue-600">
                  {count} chunks
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemStatus;
