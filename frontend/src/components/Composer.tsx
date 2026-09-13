import { useState, type KeyboardEvent } from "react";
import { Send, Square } from "lucide-react";

interface ComposerProps {
  isStreaming: boolean;
  onSend: (query: string) => void;
  onStop: () => void;
}

export function Composer({
  isStreaming,
  onSend,
  onStop
}: ComposerProps) {
  const [value, setValue] = useState("");

  const submit = () => {
    const query = value.trim();
    if (!query || isStreaming) {
      return;
    }

    setValue("");
    onSend(query);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入你的问题"
          rows={3}
          disabled={isStreaming}
          aria-label="输入你的问题"
        />

        {isStreaming ? (
          <button
            className="stop-button"
            type="button"
            onClick={onStop}
            title="停止生成"
            aria-label="停止生成"
          >
            <Square size={16} fill="currentColor" aria-hidden="true" />
          </button>
        ) : (
          <button
            className="send-button"
            type="button"
            onClick={submit}
            disabled={!value.trim()}
            title="发送消息"
            aria-label="发送消息"
          >
            <Send size={17} aria-hidden="true" />
          </button>
        )}
      </div>
    </div>
  );
}
