import { useEffect, useMemo, useRef, useState } from "react";
import { getHealth, streamChat } from "../api/chat";
import type {
  ChatMessage,
  Conversation,
  ServiceHealth,
  StreamSource
} from "../types/api";

interface ChatState {
  conversations: Conversation[];
  activeId: string;
}

const STORAGE_KEY = "clouddesk_chat_state_v1";

function createId(): string {
  return window.crypto.randomUUID();
}

function createEmptyConversation(): Conversation {
  return {
    id: createId(),
    title: "新会话",
    messages: [],
    updatedAt: Date.now()
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isStreamSource(value: unknown): value is StreamSource {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value.document_id === "string" &&
    typeof value.title === "string" &&
    typeof value.category === "string" &&
    (value.score === undefined || typeof value.score === "number")
  );
}

function isChatMessage(value: unknown): value is ChatMessage {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value.id === "string" &&
    (value.role === "user" || value.role === "assistant") &&
    typeof value.content === "string" &&
    Array.isArray(value.sources) &&
    value.sources.every(isStreamSource) &&
    (value.status === "streaming" ||
      value.status === "complete" ||
      value.status === "error") &&
    (value.error === undefined || typeof value.error === "string")
  );
}

function isConversation(value: unknown): value is Conversation {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value.id === "string" &&
    typeof value.title === "string" &&
    typeof value.updatedAt === "number" &&
    Array.isArray(value.messages) &&
    value.messages.every(isChatMessage)
  );
}

function loadChatState(): ChatState {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);

      if (isRecord(parsed) && Array.isArray(parsed.conversations)) {
        const conversations = parsed.conversations
          .filter(isConversation)
          .map((conversation) => ({
            ...conversation,
            messages: conversation.messages.map((message) =>
              message.status === "streaming"
                ? {
                    ...message,
                    status: "error" as const,
                    error: "上次生成被中断"
                  }
                : message
            )
          }));

        if (conversations.length > 0) {
          const storedActiveId =
            typeof parsed.activeId === "string" ? parsed.activeId : "";
          const activeId = conversations.some(
            (conversation) => conversation.id === storedActiveId
          )
            ? storedActiveId
            : conversations[0].id;

          return { conversations, activeId };
        }
      }
    }
  } catch {
    // 本地缓存损坏时直接创建新会话，不影响主流程。
  }

  const conversation = createEmptyConversation();
  return { conversations: [conversation], activeId: conversation.id };
}

function createTitle(query: string): string {
  const compact = query.replace(/\s+/g, " ").trim();
  return compact.length > 22 ? `${compact.slice(0, 22)}...` : compact;
}

export function useChat() {
  const [chatState, setChatState] = useState<ChatState>(loadChatState);
  const [isStreaming, setIsStreaming] = useState(false);
  const [health, setHealth] = useState<ServiceHealth>("checking");
  const abortRef = useRef<AbortController | null>(null);

  const activeConversation = useMemo(
    () =>
      chatState.conversations.find(
        (conversation) => conversation.id === chatState.activeId
      ) ?? null,
    [chatState]
  );

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(chatState));
  }, [chatState]);

  useEffect(() => {
    let cancelled = false;

    const checkHealth = async () => {
      try {
        await getHealth();
        if (!cancelled) {
          setHealth("online");
        }
      } catch {
        if (!cancelled) {
          setHealth("offline");
        }
      }
    };

    void checkHealth();
    const timer = window.setInterval(() => void checkHealth(), 30000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const patchMessage = (
    conversationId: string,
    messageId: string,
    patch: Partial<ChatMessage>
  ) => {
    setChatState((current) => ({
      ...current,
      conversations: current.conversations.map((conversation) =>
        conversation.id === conversationId
          ? {
              ...conversation,
              messages: conversation.messages.map((message) =>
                message.id === messageId
                  ? { ...message, ...patch }
                  : message
              ),
              updatedAt: Date.now()
            }
          : conversation
      )
    }));
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
  };

  const createConversation = () => {
    stopGeneration();
    const conversation = createEmptyConversation();

    setChatState((current) => ({
      conversations: [conversation, ...current.conversations],
      activeId: conversation.id
    }));
  };

  const selectConversation = (id: string) => {
    stopGeneration();
    setChatState((current) => ({
      ...current,
      activeId: id
    }));
  };

  const sendMessage = async (query: string) => {
    const trimmedQuery = query.trim();
    const conversation = activeConversation;

    if (!trimmedQuery || !conversation || isStreaming) {
      return;
    }

    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: trimmedQuery,
      sources: [],
      status: "complete"
    };
    const assistantMessage: ChatMessage = {
      id: createId(),
      role: "assistant",
      content: "",
      sources: [],
      status: "streaming"
    };
    const title =
      conversation.messages.length === 0
        ? createTitle(trimmedQuery)
        : conversation.title;

    setChatState((current) => ({
      ...current,
      conversations: current.conversations.map((item) =>
        item.id === conversation.id
          ? {
              ...item,
              title,
              messages: [
                ...item.messages,
                userMessage,
                assistantMessage
              ],
              updatedAt: Date.now()
            }
          : item
      )
    }));

    const controller = new AbortController();
    abortRef.current = controller;
    setIsStreaming(true);
    let terminalEventReceived = false;

    try {
      await streamChat(
        { query: trimmedQuery },
        {
          onSources: (sources) => {
            patchMessage(conversation.id, assistantMessage.id, { sources });
          },
          onToken: (content) => {
            setChatState((current) => ({
              ...current,
              conversations: current.conversations.map((item) =>
                item.id === conversation.id
                  ? {
                      ...item,
                      messages: item.messages.map((message) =>
                        message.id === assistantMessage.id
                          ? { ...message, content: message.content + content }
                          : message
                      ),
                      updatedAt: Date.now()
                    }
                  : item
              )
            }));
          },
          onDone: () => {
            terminalEventReceived = true;
            patchMessage(conversation.id, assistantMessage.id, {
              status: "complete"
            });
          },
          onError: (message) => {
            terminalEventReceived = true;
            patchMessage(conversation.id, assistantMessage.id, {
              status: "error",
              error: message
            });
          }
        },
        controller.signal
      );

      if (!terminalEventReceived) {
        throw new Error("流式连接提前结束");
      }
    } catch (error) {
      const isAbortError =
        error instanceof DOMException && error.name === "AbortError";
      const message = isAbortError
        ? "已停止生成"
        : error instanceof Error
          ? error.message
          : "请求失败，请稍后重试";

      patchMessage(conversation.id, assistantMessage.id, {
        status: "error",
        error: message
      });
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      setIsStreaming(false);
    }
  };

  return {
    conversations: chatState.conversations,
    activeConversation,
    isStreaming,
    health,
    createConversation,
    selectConversation,
    sendMessage,
    stopGeneration
  };
}
