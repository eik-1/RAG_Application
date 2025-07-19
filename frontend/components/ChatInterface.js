"use client";

import { useState, useRef, useEffect } from "react";
import {
  Send,
  RefreshCw,
  Trash2,
  FileText,
  Copy,
  ExternalLink,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import { apiService } from "../lib/api";

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showSources, setShowSources] = useState(true);
  const [error, setError] = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!currentMessage.trim() || isLoading) return;

    const userMessage = currentMessage;
    setCurrentMessage("");
    setError(null);

    // Add user message to chat
    const messageId = Date.now().toString();
    const newMessage = {
      id: messageId,
      message: userMessage,
      response: "",
      timestamp: new Date(),
      isLoading: true,
    };

    setMessages((prev) => [...prev, newMessage]);
    setIsLoading(true);

    try {
      const request = {
        message: userMessage,
        include_sources: showSources,
        top_k: 5,
      };

      const response = await apiService.sendMessage(request);

      // Update message with response
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId
            ? {
                ...msg,
                response: response.response,
                sources: response.sources,
                retrievalStats: response.retrieval_stats,
                isLoading: false,
              }
            : msg
        )
      );
    } catch (error) {
      console.error("Failed to send message:", error);
      setError("Failed to send message. Please try again.");

      // Update message with error
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId
            ? {
                ...msg,
                response:
                  "I apologize, but I encountered an error processing your message. Please try again.",
                isLoading: false,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = async () => {
    try {
      await apiService.clearMemory();
      setMessages([]);
      setError(null);
    } catch (error) {
      console.error("Failed to clear chat:", error);
      setError("Failed to clear conversation memory.");
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  const formatDocumentName = (docName) => {
    if (!docName) return "Unknown Source";
    return docName.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] bg-white rounded-lg shadow-lg">
      {/* Chat Header */}
      <div className="flex justify-between items-center p-4 border-b bg-gray-50 rounded-t-lg">
        <div>
          <h2 className="text-lg font-semibold text-gray-800">
            RAG Chat Interface
          </h2>
          <p className="text-sm text-gray-600">
            Ask questions about the research papers and get contextual answers
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <label className="flex items-center space-x-2 text-sm">
            <input
              type="checkbox"
              checked={showSources}
              onChange={(e) => setShowSources(e.target.checked)}
              className="rounded"
            />
            <span>Show Sources</span>
          </label>

          <button
            onClick={clearChat}
            className="p-2 text-gray-500 hover:text-red-600 transition-colors"
            title="Clear conversation"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 border-b border-red-200">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-20">
            <FileText className="w-12 h-12 mx-auto mb-4 text-gray-300" />
            <h3 className="text-lg font-medium mb-2">Welcome to RAG Chat</h3>
            <p className="text-sm">
              Start a conversation by asking questions about the research
              papers.
              <br />
              Try asking about transformers, BERT, GPT-3, or other NLP topics.
            </p>

            <div className="mt-6 space-y-2 text-left max-w-md mx-auto">
              <p className="text-xs font-medium text-gray-600 mb-2">
                Example questions:
              </p>
              <button
                onClick={() =>
                  setCurrentMessage("What is the transformer architecture?")
                }
                className="block w-full text-left p-2 text-xs bg-blue-50 hover:bg-blue-100 rounded transition-colors"
              >
                &ldquo;What is the transformer architecture?&rdquo;
              </button>
              <button
                onClick={() =>
                  setCurrentMessage("How does BERT differ from GPT?")
                }
                className="block w-full text-left p-2 text-xs bg-blue-50 hover:bg-blue-100 rounded transition-colors"
              >
                &ldquo;How does BERT differ from GPT?&rdquo;
              </button>
              <button
                onClick={() =>
                  setCurrentMessage("What are the key innovations in T5?")
                }
                className="block w-full text-left p-2 text-xs bg-blue-50 hover:bg-blue-100 rounded transition-colors"
              >
                &ldquo;What are the key innovations in T5?&rdquo;
              </button>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div key={message.id} className="space-y-4">
                {/* User Message */}
                <div className="flex justify-end">
                  <div className="max-w-3xl bg-blue-600 text-white p-3 rounded-lg">
                    <div className="whitespace-pre-wrap">{message.message}</div>
                  </div>
                </div>

                {/* Assistant Response */}
                <div className="flex justify-start">
                  <div className="max-w-4xl w-full">
                    <div className="bg-gray-50 p-4 rounded-lg">
                      {message.isLoading ? (
                        <div className="flex items-center space-x-2">
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          <span className="text-gray-600">Thinking...</span>
                        </div>
                      ) : (
                        <>
                          <div className="prose prose-sm max-w-none">
                            <ReactMarkdown>{message.response}</ReactMarkdown>
                          </div>

                          <div className="mt-3 flex items-center justify-between">
                            <button
                              onClick={() => copyToClipboard(message.response)}
                              className="text-xs text-gray-500 hover:text-gray-700 flex items-center space-x-1"
                            >
                              <Copy className="w-3 h-3" />
                              <span>Copy</span>
                            </button>

                            {message.retrievalStats && (
                              <div className="text-xs text-gray-500">
                                {message.retrievalStats.chunks_found} chunks •
                                Score:{" "}
                                {message.retrievalStats.top_score.toFixed(3)}
                              </div>
                            )}
                          </div>
                        </>
                      )}
                    </div>

                    {/* Sources */}
                    {!message.isLoading &&
                      message.sources &&
                      message.sources.length > 0 &&
                      showSources && (
                        <div className="mt-3 bg-blue-50 p-3 rounded-lg">
                          <h4 className="text-sm font-medium text-blue-800 mb-2 flex items-center">
                            <ExternalLink className="w-3 h-3 mr-1" />
                            Sources ({message.sources.length})
                          </h4>
                          <div className="space-y-2">
                            {message.sources.map((source, index) => (
                              <div
                                key={index}
                                className="bg-white p-2 rounded border-l-2 border-blue-300"
                              >
                                <div className="flex justify-between items-start mb-1">
                                  <span className="text-xs font-medium text-blue-700">
                                    {formatDocumentName(source.source)}
                                  </span>
                                  <span className="text-xs text-gray-500">
                                    {source.similarity_score ? source.similarity_score.toFixed(3) : "N/A"}
                                  </span>
                                </div>
                                <p className="text-xs text-gray-600 leading-relaxed">
                                  {source.text_preview || "No preview available"}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                  </div>
                </div>
              </div>
            ))}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t bg-gray-50 p-4 rounded-b-lg">
        <div className="flex space-x-2">
          <textarea
            ref={inputRef}
            value={currentMessage}
            onChange={(e) => setCurrentMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about the research papers..."
            className="flex-1 resize-none border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={2}
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={!currentMessage.trim() || isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            {isLoading ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            <span>Send</span>
          </button>
        </div>

        <div className="mt-2 text-xs text-gray-500 text-center">
          Press Enter to send • Shift+Enter for new line • {messages.length}{" "}
          messages in conversation
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
