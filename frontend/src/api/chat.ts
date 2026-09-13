import type {
  ChatRequest,
  DoneEventData,
  HealthResponse,
  StreamCallbacks,
  StreamSource
} from "../types/api";

interface ParsedSseFrame {
  event: string;
  data: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readString(
  value: Record<string, unknown>,
  key: string
): string | undefined {
  return typeof value[key] === "string" ? value[key] : undefined;
}

function readNumber(
  value: Record<string, unknown>,
  key: string
): number | undefined {
  return typeof value[key] === "number" ? value[key] : undefined;
}

function parseSources(data: unknown): StreamSource[] {
  if (!Array.isArray(data)) {
    throw new Error("sources 事件格式不正确");
  }

  return data.map((item) => {
    if (!isRecord(item)) {
      throw new Error("source 数据格式不正确");
    }

    const documentId = readString(item, "document_id");
    const title = readString(item, "title");
    const category = readString(item, "category");

    if (!documentId || !title || !category) {
      throw new Error("source 缺少必要字段");
    }

    return {
      document_id: documentId,
      title,
      category,
      score: readNumber(item, "score")
    };
  });
}

function parseToken(data: unknown): string {
  if (!isRecord(data) || typeof data.content !== "string") {
    throw new Error("token 事件格式不正确");
  }

  return data.content;
}

function parseDone(data: unknown): DoneEventData {
  if (!isRecord(data) || typeof data.status !== "string") {
    throw new Error("done 事件格式不正确");
  }

  return {
    status: data.status,
    latency_ms: readNumber(data, "latency_ms"),
    vector_count: readNumber(data, "vector_count"),
    keyword_count: readNumber(data, "keyword_count"),
    rrf_top_k: readNumber(data, "rrf_top_k"),
    rewritten_query: readString(data, "rewritten_query"),
    query_category: readString(data, "query_category")
  };
}

function parseError(data: unknown): string {
  if (!isRecord(data) || typeof data.message !== "string") {
    return "流式请求发生未知错误";
  }

  return data.message;
}

function parseFrame(rawFrame: string): ParsedSseFrame {
  const lines = rawFrame.split("\n");
  const dataLines: string[] = [];
  let event = "message";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) {
    throw new Error("SSE 事件缺少 data 字段");
  }

  return {
    event,
    data: JSON.parse(dataLines.join("\n")) as unknown
  };
}

function dispatchFrame(
  rawFrame: string,
  callbacks: StreamCallbacks
): boolean {
  const frame = parseFrame(rawFrame);

  switch (frame.event) {
    case "sources":
      callbacks.onSources(parseSources(frame.data));
      return true;
    case "token":
      callbacks.onToken(parseToken(frame.data));
      return true;
    case "done":
      callbacks.onDone(parseDone(frame.data));
      return false;
    case "error":
      callbacks.onError(parseError(frame.data));
      return false;
    default:
      return true;
  }
}

// POST SSE 不能使用 EventSource，因此通过 fetch ReadableStream 手动读取事件块。
export async function streamChat(
  request: ChatRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch("/api/v1/chat/stream", {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json"
    },
    body: JSON.stringify(request),
    signal
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `聊天请求失败 (${response.status})${detail ? `: ${detail}` : ""}`
    );
  }

  if (!response.body) {
    throw new Error("浏览器未收到可读取的流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const rawFrame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        if (rawFrame.trim()) {
          const shouldContinue = dispatchFrame(rawFrame, callbacks);
          if (!shouldContinue) {
            return;
          }
        }

        boundary = buffer.indexOf("\n\n");
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      dispatchFrame(buffer, callbacks);
    }
  } finally {
    reader.releaseLock();
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch("/api/v1/health", {
    headers: { Accept: "application/json" }
  });

  if (!response.ok) {
    throw new Error(`健康检查失败 (${response.status})`);
  }

  const data: unknown = await response.json();
  if (!isRecord(data) || typeof data.status !== "string") {
    throw new Error("健康检查返回格式不正确");
  }

  const rawServices = isRecord(data.services) ? data.services : {};
  const services: Record<string, string> = {};

  for (const [name, value] of Object.entries(rawServices)) {
    if (typeof value === "string") {
      services[name] = value;
    }
  }

  return { status: data.status, services };
}
