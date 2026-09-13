import { AlertCircle, Bot, UserRound } from "lucide-react";
import type { ChatMessage } from "../types/api";
import { SourcesPanel } from "./SourcesPanel";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isAssistant = message.role === "assistant";

  return (
    <article
      className={
        isAssistant
          ? "message-row assistant"
          : "message-row user"
      }
    >
      <div className="message-avatar" aria-hidden="true">
        {isAssistant ? <Bot size={18} /> : <UserRound size={18} />}
      </div>

      <div className="message-content">
        <div className="message-author">
          {isAssistant ? "CloudDesk" : "你"}
        </div>

        <div className="message-bubble">
          {message.content ? (
            <p className="message-text">{message.content}</p>
          ) : message.status === "streaming" ? (
            <span className="typing-indicator" aria-label="正在生成回答">
              <i />
              <i />
              <i />
            </span>
          ) : null}
        </div>

        {message.error ? (
          <div className="message-error" role="alert">
            <AlertCircle size={15} aria-hidden="true" />
            <span>{message.error}</span>
          </div>
        ) : null}

        {isAssistant && message.sources.length > 0 ? (
          <SourcesPanel sources={message.sources} />
        ) : null}
      </div>
    </article>
  );
}
