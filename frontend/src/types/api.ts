// 与后端真实返回结构对应的 TypeScript 类型。
export interface ChatRequest {
  query: string;
  session_id?: string;
  top_k?: number;
}

export interface StreamSource {
  document_id: string;
  title: string;
  category: string;
  score?: number;
}

export interface TokenEventData {
  content: string;
}

export interface DoneEventData {
  status: string;
  latency_ms?: number;
  vector_count?: number;
  keyword_count?: number;
  rrf_top_k?: number;
  rewritten_query?: string;
  query_category?: string;
}

export interface StreamErrorData {
  message: string;
}

export type SSEEventData =
  | StreamSource[]
  | TokenEventData
  | DoneEventData
  | StreamErrorData;

export interface StreamCallbacks {
  onSources: (sources: StreamSource[]) => void;
  onToken: (content: string) => void;
  onDone: (data: DoneEventData) => void;
  onError: (message: string) => void;
}

export type MessageStatus = "streaming" | "complete" | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: StreamSource[];
  status: MessageStatus;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  updatedAt: number;
}

export interface HealthResponse {
  status: string;
  services: Record<string, string>;
}

export type ServiceHealth = "checking" | "online" | "offline";
