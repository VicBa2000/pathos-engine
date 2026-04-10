import { Component, type ReactNode, type ErrorInfo } from "react";

interface Props {
  children: ReactNode;
  fallbackLabel?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[ErrorBoundary] ${error.message}`, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      const label = this.props.fallbackLabel ?? "This panel";
      return (
        <div style={{
          padding: "1.5rem",
          textAlign: "center",
          color: "#999",
          fontSize: "0.85rem",
        }}>
          <p style={{ marginBottom: "0.5rem" }}>
            {label} encountered an error and was reset.
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              padding: "0.4rem 1rem",
              background: "#333",
              color: "#ccc",
              border: "1px solid #555",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "0.8rem",
            }}
          >
            Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
