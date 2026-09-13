import { BookOpenText } from "lucide-react";
import type { StreamSource } from "../types/api";

interface SourcesPanelProps {
  sources: StreamSource[];
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  return (
    <details className="sources-panel" open>
      <summary>
        <BookOpenText size={15} aria-hidden="true" />
        <span>引用来源</span>
        <span className="source-count">{sources.length}</span>
      </summary>

      <div className="source-list">
        {sources.map((source, index) => (
          <div
            className="source-item"
            key={`${source.document_id}-${index}`}
          >
            <div className="source-title">{source.title}</div>
            <div className="source-meta">
              <span className="source-category">{source.category}</span>
              <code>{source.document_id}</code>
              {source.score !== undefined ? (
                <span className="source-score">
                  score {source.score.toFixed(4)}
                </span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}
