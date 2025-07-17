"use client";

import { useState, useEffect } from "react";
import { BarChart3, FileText, RefreshCw } from "lucide-react";
import { apiService } from "../lib/api";

const DocumentStats = () => {
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const documentStats = await apiService.getDocumentStats();
      setStats(documentStats);
    } catch (error) {
      console.error("Failed to load document stats:", error);
      setError("Failed to load document statistics");
    } finally {
      setIsLoading(false);
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatDocumentName = (docName) => {
    return docName.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center space-x-2">
          <RefreshCw className="w-5 h-5 animate-spin" />
          <span>Loading document statistics...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-2">⚠️</div>
          <p className="text-red-600">{error}</p>
          <button
            onClick={loadStats}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-center text-gray-500">
          <FileText className="w-12 h-12 mx-auto mb-4 text-gray-300" />
          <p>No document statistics available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <BarChart3 className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">
              Document Statistics
            </h2>
          </div>
          <button
            onClick={loadStats}
            className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
            title="Refresh statistics"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600">
              {stats.total_documents}
            </div>
            <div className="text-sm text-gray-600 mt-1">Total Documents</div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600">
              {stats.total_chunks.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600 mt-1">Total Chunks</div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600">
              {formatBytes(stats.total_characters)}
            </div>
            <div className="text-sm text-gray-600 mt-1">Total Content</div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-orange-600">
              {Math.round(stats.avg_chunk_size)}
            </div>
            <div className="text-sm text-gray-600 mt-1">Avg Chunk Size</div>
          </div>
        </div>
      </div>

      {/* Document Details */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Document Breakdown
        </h3>

        <div className="space-y-4">
          {Object.entries(stats.documents).map(([docName, docStats]) => (
            <div
              key={docName}
              className="border border-gray-200 rounded-lg p-4"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h4 className="font-medium text-gray-900">
                    {formatDocumentName(docName)}
                  </h4>
                  <p className="text-sm text-gray-500">{docName}</p>
                </div>
                <div className="text-right">
                  <div className="text-lg font-semibold text-blue-600">
                    {docStats.chunks} chunks
                  </div>
                  <div className="text-sm text-gray-500">
                    {((docStats.chunks / stats.total_chunks) * 100).toFixed(1)}%
                    of total
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Total Characters:</span>
                  <div className="font-medium">
                    {docStats.total_characters.toLocaleString()}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Content Size:</span>
                  <div className="font-medium">
                    {formatBytes(docStats.total_characters)}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Avg Chunk Size:</span>
                  <div className="font-medium">
                    {Math.round(docStats.avg_chunk_size)} chars
                  </div>
                </div>
              </div>

              {/* Progress bar for chunks */}
              <div className="mt-3">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{
                      width: `${(docStats.chunks / stats.total_chunks) * 100}%`,
                    }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Charts placeholder - could be enhanced with actual charts */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Content Distribution
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Chunk distribution */}
          <div>
            <h4 className="text-md font-medium text-gray-700 mb-3">
              Chunks per Document
            </h4>
            <div className="space-y-2">
              {Object.entries(stats.documents)
                .sort(([, a], [, b]) => b.chunks - a.chunks)
                .map(([docName, docStats]) => (
                  <div
                    key={docName}
                    className="flex items-center justify-between"
                  >
                    <span
                      className="text-sm truncate max-w-[200px]"
                      title={formatDocumentName(docName)}
                    >
                      {formatDocumentName(docName)}
                    </span>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-green-500 h-2 rounded-full"
                          style={{
                            width: `${
                              (docStats.chunks /
                                Math.max(
                                  ...Object.values(stats.documents).map(
                                    (d) => d.chunks
                                  )
                                )) *
                              100
                            }%`,
                          }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-gray-600 w-12 text-right">
                        {docStats.chunks}
                      </span>
                    </div>
                  </div>
                ))}
            </div>
          </div>

          {/* Size distribution */}
          <div>
            <h4 className="text-md font-medium text-gray-700 mb-3">
              Content Size per Document
            </h4>
            <div className="space-y-2">
              {Object.entries(stats.documents)
                .sort(([, a], [, b]) => b.total_characters - a.total_characters)
                .map(([docName, docStats]) => (
                  <div
                    key={docName}
                    className="flex items-center justify-between"
                  >
                    <span
                      className="text-sm truncate max-w-[200px]"
                      title={formatDocumentName(docName)}
                    >
                      {formatDocumentName(docName)}
                    </span>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-purple-500 h-2 rounded-full"
                          style={{
                            width: `${
                              (docStats.total_characters /
                                Math.max(
                                  ...Object.values(stats.documents).map(
                                    (d) => d.total_characters
                                  )
                                )) *
                              100
                            }%`,
                          }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-gray-600 w-16 text-right">
                        {formatBytes(docStats.total_characters)}
                      </span>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentStats;
